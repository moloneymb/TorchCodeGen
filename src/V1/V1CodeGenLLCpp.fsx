// TODO Missing roughly 50% of operators and we're not sure why yet
(*
#load "CodeGenBase.fsx"
open Clippy
open TorchMetadataParser
open TorchMetadataExtensions
open DiffPlex
open System
open System.IO
open TargetModuleCpp
open CodeGenBase

// NOTE Dynamic type is if there is an array type
// NOTE Rust generator doesn't use 'secondName' instead numbers the low level interface

// Func.name; args; returns; kind
// TODO remove deprication

// Array BaseType [|Dimname; Tensor; Int; Bool|]
//let allInputParams = schemas |> Array.collect (fun x -> x.inputs)
//let allOutputParams = schemas |> Array.collect (fun x -> x.outputs)

// 14 output arrays
//allOutputParams |> Array.filter (fun x -> x.IsArray) |> Array.length


// TODO fixed of int, dynamic
//
//// All empty input name params are Atrix
//allInputParams |> Array.filter (fun x -> x.name = "" && x.baseType <> BaseType.Astrix)
//
//// 1468 out of 1614 outputs don't have names
////(allOutputParams |> Array.filter (fun x -> x.name.IsNone && x.baseType <> BaseType.Astrix) ).Length


//// Output masks have arrays
//allParams |> Array.filter (fun x -> x.baseType = BaseType.Bool && x.array.IsSome) |> Array.map (fun x -> x.name)
//// names, indices optional array for Dimname
//allParams |> Array.filter (fun x -> x.array.IsSome && x.optional) |> Array.map (fun x -> x.name)
//// Only Dimname, Tensor, Int, Bool are arrays
////allParams |> Array.filter (fun x -> x.array.IsSome) |> Array.map (fun x -> x.baseType) |> Array.distinct
////allParams |> Array.filter (fun x -> x.optional) |> Array.map (fun x -> x.baseType) |> Array.distinct

// Any defaultValue w/o optional
//allParams |> Array.filter (fun x -> x.defaultValue.IsSome && not x.optional) |> Array.map (fun x -> x.name) |> 

// NOTE: `function vs `method has to do with the number of tensors returned
let filterParams (xs: ParamType[]) = 
    xs
    |> Array.filter (fun x -> 
        match x.IsArray, x.baseType with 
        | true, BaseType.Bool
        | _, BaseType.Astrix 
        | _, BaseType.MemoryFormat
        //| _, BaseType.Int
        | _, BaseType.Generator
        | _, BaseType.Dimname 
        | _, BaseType.Layout
        | _, BaseType.Storage
        | _, BaseType.ConstQuantizerPtr
        | _, BaseType.String
        | true, BaseType.Float -> false 
        | _ -> true)

module Func = 
    //linear.inputs.[0].modifiers
    let cTypedArgsList (xs: ParamType[]) =
        let xs =  filterParams xs
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
                    | _ -> failwithf "err %A" x
                else
                    match x.baseType with
                    | BaseType.Bool -> "int"
                    | BaseType.Int-> "int64_t"
                    | BaseType.Float -> "double"
                    | BaseType.Tensor -> "tensor"
                    | BaseType.ScalarType -> "int"
                    | BaseType.Device -> "int"
                    | BaseType.Scalar -> "scalar"
                    | _ -> failwithf "err %A" x
                    |> fun x -> sprintf "%s %s" x name 
        |] |> String.concat ", "

    let cArgsList (args: ParamType[]) = 
        let args = filterParams args
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
                    | _ -> failwithf "err %A" x
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
                    | _ -> failwithf "err %A" x
        |] |> String.concat ", "

    let cCall (x: Func) = 
        match x with
        | Function(x) -> sprintf "torch::%s(%s)" x.firstName (cArgsList x.inputs)
        | Method(x) -> 
            match x.inputs  with
            | [||] -> failwithf "Method calls should have at least one argument %s" x.firstName
            | _ -> sprintf "%s->%s(%s)" x.inputs.[0].name x.firstName (cArgsList x.inputs.[1..])
        // TODO find a better way to figure out Function vs Mehod
        | _ -> sprintf "torch::%s(%s)" x.firstName (cArgsList x.inputs)

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


let funcs = 
    //// NOTE: Not sure how to use this
    //let noTensorOptions = set [
    //      "zeros_like";
    //      "empty_like";
    //      "full_like";
    //      "ones_like";
    //      "rand_like";
    //      "randint_like";
    //      "randn_like"; ]
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
    let excludedPrefixes = set [ "_thnn_"; "_th_"; "thnn_"; "th_" ]
    let excludedSuffixes = set [ "_forward"; "_forward_out" ]
    let methods = 
        let baseFunc : Func = 
            {firstName = ""; secondName = ""; inputs = [||]; outputs = [||]; attributes = [||]}
        let f (name: string)  (baseType: BaseType) : ParamType = 
            { baseType = baseType;  name = name; defaultValue = None; array = None; alpha = None; optional = false }
        [|
            "grad", [||]
            "set_requires_grad", [|f "r" BaseType.Bool|]
            "toType", [|f "scalar_type" BaseType.ScalarType|]
            "to", [|f "device" BaseType.Device|]
        |] |> Array.map (fun (name,inputs) -> 
            {baseFunc with firstName = name; inputs = [|yield f "self" BaseType.Tensor; yield! inputs|]})
    let schemas = 
        loadSchemas(Path.Combine(__SOURCE_DIRECTORY__, "TorchMetadata.yaml"))
        |> Array.filter (fun x -> x.secondName <> "deprecated") // filters out 2
        |> Array.filter (fun x -> excludedPrefixes |> Seq.exists (fun y -> x.firstName.StartsWith(y)) |> not) // 18
        |> Array.filter (fun x -> excludedSuffixes|> Seq.exists (fun y -> x.firstName.EndsWith(y)) |> not) // 12
        |> Array.filter (fun x -> excludedFunctions.Contains(x.firstName) |> not) // 20

    [|yield! methods; yield! schemas|]
    // TODO I'm not sure about this
    //|> Array.filter (function | Function(_) | Method(_) -> true | _ -> false)
    |> Array.groupBy (fun x -> x.firstName + match x.secondName with | "out" -> "_out" | _ -> "")
    //|> Array.groupBy (fun x -> x.firstName)
    |> Array.collect ( function | (x,[|y|]) -> [|(x,y)|] 
                                //| (_,ys) -> ys |> Array.map (fun y -> (sprintf "%s_%s" y.firstName y.secondName, y)))
                                | (_,ys) -> ys |> Array.mapi (fun i y  -> (sprintf "%s%s" y.firstName  (if i = 0 then "" else string i),y)))
    |> Array.map (fun (x,y) -> x.ToLower(),y)
    |> Map.ofArray


//funcs |> Map.toArray  |> Array.length

let codeGenCpp(funcs: Map<string,Func>, blackList: Set<string>) = 
    // Create mapping, Rust uses numbers we can use second name
    [|
        yield "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!";
        yield ""
        for KeyValue(exportedName,func) in funcs do
            printfn "%s" exportedName
            if not(blackList.Contains(exportedName)) then
                let cTypedArgsList = Func.cTypedArgsList func.inputs 
                match func.OutputSize with
                | Some(ntensors) -> 
                    yield sprintf "void atg_%s(tensor *out__, %s) {" exportedName cTypedArgsList
                    yield "  PROTECT("
                    yield sprintf "    auto outputs__ = %s;" (Func.cCall func);
                    match ntensors with
                    | 1 -> yield sprintf "    out__[0] = new torch::Tensor(outputs__);"
                    | _ -> 
                        for i = 0 to ntensors - 1 do
                            yield sprintf "    out__[%d] = new torch::Tensor(std::get<%d>(outputs__));" i i 
                        yield! [|"  )"; ";"; ""|]
                | None -> 
                    yield sprintf  "tensor *atg_%s(%s) {" exportedName cTypedArgsList
                    yield  "  PROTECT("
                    yield sprintf "    auto outputs__ = %s;" (Func.cCall func)
                    (* the returned type is a C++ vector of tensors *)
                    yield "    int sz = outputs__.size();"
                    yield "    torch::Tensor **out__ = (torch::Tensor**)malloc((sz + 1) * sizeof(torch::Tensor*));"
                    yield "    for (int i = 0; i < sz; ++i)";
                    yield "      out__[i] = new torch::Tensor(outputs__[i]);";
                    yield "    out__[sz] = nullptr;";
                    yield "    return out__;";
                    yield "  )";
                    yield "  return nullptr;";
                    yield! [|"}";""|]
    |]

let codeGenHpp(funcs: Map<string,Func>) = 
    // Create mapping, Rust uses numbers we can use second name
    [|
        yield "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!";
        yield ""
        for KeyValue(exportedName,func) in funcs do
            let cTypedArgsList = Func.cTypedArgsList func.inputs
            match func.OutputSize with
            | Some(ntensors) -> 
                yield sprintf "void atg_%s(tensor *, %s);" exportedName cTypedArgsList
            | None -> 
                yield sprintf  "tensor *atg_%s(%s);" exportedName cTypedArgsList
    |]

//let bl = set [ ]
//    "_amp_update_scale" // Float input
//    "__xor___Tensor" // Astrix
//    "__xor___Scalar" // Astrix 
//    "__rshift___Tensor"
//    "__rshift___Scalar"

//let cpp = codeGenCpp(funcs,bl) |> String.concat "\n"
//cpp

let range = (20,100)

let hpp = 
    funcs |> codeGenHpp  
    //|> Array.filter (fun x -> x.Contains("ctc_loss"))
    |> Array.sort 
    |> Array.skip (fst range) |> Array.take (snd range) 
    |> String.concat "\n"

let hpp2 = 
    File.ReadAllLines(@"C:\EE\Git\TorchCodeGen\src\Rust\torch_api_generated.h") 
    //|> Array.filter (fun x -> x.Contains("ctc_loss"))
    |> Array.sort
    |> Array.skip (fst range) |> Array.take (snd range)
    |> String.concat "\n"


hpp |> DiffPlex.showDiff hpp2 
hpp |> Clippy.copyToClipboard
hpp2 |> Clippy.copyToClipboard

// Given two string arrays find the ones with the matching hash?

//let schemas = loadSchemas(Path.Combine(__SOURCE_DIRECTORY__, "TorchMetadata.yaml"))
//
//let getSchmea(firstName,secondName) = 
//    schemas |> Array.find (fun x -> x.firstName = firstName && x.secondName = secondName)
//let searchSchemas(name: string) = 
//    schemas 
//    |> Array.filter (fun x -> x.firstName.Contains(name) || x.secondName.Contains(name)) 
//    |> Array.map (fun x -> x.firstName,x.secondName)
//
*)
