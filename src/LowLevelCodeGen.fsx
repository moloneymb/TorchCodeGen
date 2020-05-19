// TODO move CombinedModifiers into ParamType
// TODO possibly merge CombinedModifiers into BaseType to see if it matches the Rust type
#load "CodeGenBase.fsx"
open Clippy
open TorchMetadataParser
open TorchMetadataExtensions
open DiffPlex
open System
open System.IO
open TargetModuleCpp
open CodeGenBase

// Array BaseType [|Dimname; Tensor; Int; Bool|]
let allInputParams = schemas |> Array.collect (fun x -> x.inputs)
let allOutputParams = schemas |> Array.collect (fun x -> x.outputs)

// 14 output arrays
allOutputParams |> Array.filter (fun x -> x.IsArray) |> Array.length


// TODO fixed of int, dynamic
//
//// All empty input name params are Atrix
//allInputParams |> Array.filter (fun x -> x.name = "" && x.baseType <> BaseType.Astrix)
//
//// 1468 out of 1614 outputs don't have names
////(allOutputParams |> Array.filter (fun x -> x.name.IsNone && x.baseType <> BaseType.Astrix) ).Length
//
let allParams = [|yield! allInputParams; yield! allOutputParams|]

schemas |> Array.collect (fun x -> x.outputs) |> Array.distinct

//schemas |> Array.collect (fun x -> x.attributes) |> Array.filter (function | Dispatch _ -> false | _ -> true)  |> Array.distinct

//allParams |> Array.map (fun x -> x.)
//
//let xx = allParams.[0].modifiers

//// Output masks have arrays
//allParams |> Array.filter (fun x -> x.baseType = BaseType.Bool && x.array.IsSome) |> Array.map (fun x -> x.name)
//// names, indices optional array for Dimname
//allParams |> Array.filter (fun x -> x.array.IsSome && x.optional) |> Array.map (fun x -> x.name)
//// Only Dimname, Tensor, Int, Bool are arrays
////allParams |> Array.filter (fun x -> x.array.IsSome) |> Array.map (fun x -> x.baseType) |> Array.distinct
////allParams |> Array.filter (fun x -> x.optional) |> Array.map (fun x -> x.baseType) |> Array.distinct

// Any defaultValue w/o optional
//allParams |> Array.filter (fun x -> x.defaultValue.IsSome && not x.optional) |> Array.map (fun x -> x.name) |> 


//  type arg_type =
//    | Bool
//    | Int64
//    | Double
//    | Tensor
//    | TensorOption
//    | IntList
//    | TensorList
//    | TensorOptions
//    | Scalar
//    | ScalarType
//    | Device
//    | String
//
// schemas.Length // 1368

// See gen.ml

let excludedFunctions  = set [
      "multi_margin_loss";
      "multi_margin_loss_out";
      "log_softmax_backward_data";
      "softmax_backward_data";
      "clone";
      "copy_";
      "conv_transpose2d_backward_out";
      "conv_transpose3d_backward_out";
      "slow_conv_transpose2d_backward_out";
      "slow_conv_transpose3d_backward_out";
      "slow_conv3d_backward_out";
      "normal";
      "_cufft_set_plan_cache_max_size";
      "_cufft_clear_plan_cache";
      "backward";
      "set_data";
      "_amp_non_finite_check_and_unscale_";
      "_cummin_helper";
      "_cummax_helper";
      "retain_grad"; ]

let noTensorOptions = set [
      "zeros_like";
      "empty_like";
      "full_like";
      "ones_like";
      "rand_like";
      "randint_like";
      "randn_like"; ]

let prefixedFunctions = 
    set [ "add"; "add_"; "div"; "div_"; "mul"; "mul_";
          "sub"; "sub_"; "nll_loss" ]

let excludedPrefixes = set [ "_thnn_"; "_th_"; "thnn_"; "th_" ]

let excluded_suffixes = set [ "_forward"; "_forward_out" ]

let linear = getSchmea("linear","")

module Func = 
    //linear.inputs.[0].modifiers
    let cTypedArgsList (xs: ParamType[]) =
        [|
            for x in xs do
                let name = x.name
                if x.IsArray then
                    match x.baseType with
                    | BaseType.Bool -> failwith "todo"
                    | BaseType.Int -> sprintf "int64_t *%s_data, int %s_len" name name
                    | BaseType.Tensor when x.optional -> sprintf "int %s_kind, int %s_device" name name
                    | BaseType.Tensor when not x.optional -> sprintf "tensor *%s_data, int %s_len" name name
                    | BaseType.Dimname -> failwith "todo"
                    //| BaseType.Strig -> 
                    //    sprintf "char* %s_ptr, int %s_len" name name
                    | _ -> failwith "err"
                else
                    match x.baseType with
                    | BaseType.Bool -> "int"
                    | BaseType.Int-> "int64_t"
                    | BaseType.Float -> "double"
                    | BaseType.Tensor -> "tensor"
                    | BaseType.ScalarType -> "int"
                    | BaseType.Device -> "int"
                    | BaseType.Scalar -> "scalar"
                    | _ -> failwith "err"
                    |> fun x -> sprintf "%s %s" x name 
        |] |> String.concat ", "

    let cArgsList (args: ParamType[]) = 
        [|
            for x in args do
                if x.IsArray then
                    match x.baseType with
                    | BaseType.Bool -> failwith "todo"
                    | BaseType.Int -> sprintf "torch::IntArrayRef(%s_data, %s_len)" x.name x.name
                    | BaseType.Tensor ->
                        if x.optional 
                        then sprintf "at::device(device_of_int(%s_device)).dtype (at::ScalarType(%s_kind))" x.name x.name 
                        else sprintf "of_carray_tensor(%s_data, %s_len)" x.name x.name
                    | BaseType.Dimname -> failwith "todo"
                    | _ -> failwith "err"
                else 
                    match x.baseType with
                    | BaseType.Bool ->  "(bool)" + x.name
                    | BaseType.Scalar ->  "*" + x.name
                    | BaseType.Tensor -> 
                        if x.optional 
                        then sprintf "(%s ? *%s : torch::Tensor())" x.name x.name  
                        else "*" + x.name 
                    | BaseType.ScalarType -> sprintf "at::ScalarType(%s)" x.name
                    | BaseType.Device -> sprintf "device_of_int(%s)" x.name
                    | BaseType.String ->  sprintf "std::string(%s_ptr, %s_len)"  x.name x.name 
                    | _ -> failwith "err"
        |] |> String.concat ", "

    let cCall (x: Func) = 
        match x with
        | Function(x) -> sprintf "torch::%s(%s)" x.firstName (cArgsList x.inputs)
        | Method(x) -> 
            match x.inputs  with
            | [||] -> failwithf "Method calls should have at least one argument %s" x.firstName
            | _ -> sprintf "%s->%s(%s)" x.inputs.[0].name x.firstName (cArgsList x.inputs.[1..])
        | _ -> failwith "err"

    let selfName = "self"
    let inputName = "input"

    let selfTensor (arg: ParamType) = 
        match arg.baseType with
        | BaseType.Tensor -> arg.name = selfName
        | _ -> false

    let inputTensor (arg: ParamType) = 
        match arg.baseType with
        | BaseType.Tensor -> arg.name = inputName
        | _ -> false
        
    // Might be Rust specific
    let typeParameters (t: Func) = 
        let needsScalarParameter = 
            t.inputs |> Array.exists (fun x -> x.baseType = BaseType.Scalar)
        let needsTypeParameter = 
            t.inputs |> Array.exists (fun x -> 
                x.baseType = BaseType.Tensor && (x.IsArray || x.optional))
        match needsTypeParameter, needsScalarParameter with
        | true,true -> "<T: Borrow<Tensor>, S: Into<Scalar>>"
        | true,false -> "<T: Borrow<Tensor>>"
        | false,true -> "<S: Into<Scalar>>"
        | false,false -> ""
        
let codeGenCpp() = 
    [|
        yield "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!";
        yield ""
        for func in schemas do
            let cTypedArgList = Func.cTypedArgsList func.inputs
            if func.IsFixedOutput then
                yield "TODO"
//                  pc "void atg_%s(tensor *out__, %s) {" exported_name
//                    c_typed_args_list;
//                  pc "  PROTECT(";
//                  pc "    auto outputs__ = %s;" (Func.c_call func);
//                  if ntensors = 1 then
//                    pc "    out__[0] = new torch::Tensor(outputs__);"
//                  else
//                    for i = 0 to ntensors - 1 do
//                      pc
//                        "    out__[%d] = new \
//                         torch::Tensor(std::get<%d>(outputs__));"
//                        i i
//                    done;
//                  pc "  )";
//                  pc "}";
//                  pc "";
            else // Dunamic
                yield "TODO"
//                    c_typed_args_list
//                  pc "tensor *atg_%s(%s) {" exported_name c_typed_args_list;
//                  pc "  PROTECT(";
//                  pc "    auto outputs__ = %s;" (Func.c_call func);
//                  (* the returned type is a C++ vector of tensors *)
//                  pc "    int sz = outputs__.size();";
//                  pc
//                    "    torch::Tensor **out__ = (torch::Tensor**)malloc((sz + \
//                     1) * sizeof(torch::Tensor*));";
//                  pc "    for (int i = 0; i < sz; ++i)";
//                  pc "      out__[i] = new torch::Tensor(outputs__[i]);";
//                  pc "    out__[sz] = nullptr;";
//                  pc "    return out__;";
//                  pc "  )";
//                  pc "  return nullptr;";
//                  pc "}";
//                  pc "";
                
            //match 
                // exported_name


    |]
