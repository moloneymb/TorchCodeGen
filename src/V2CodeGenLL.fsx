#load "V2Parser.fsx"
#load "DiffPlex.fsx"
#load "Clippy.fsx"
#load "TargetModuleCPP.fsx"
open System
open System.IO
open CodeGenCommon
open V2Parser
open TargetModuleCpp
open Clippy
open DiffPlex
open CSharp
open Cpp

let noTensorOptions =
    set [ "zeros_like"; "empty_like"; "full_like"; "ones_like"; "rand_like"; "randint_like"; "randn_like"; ]

// NOTES: 
// convolution_backward_overridable


// NOTES:
let filterParams (name: string) (arg: Arg) = 
    let f(arg: Arg) =
        match arg.type_ with
        | BT.Int
        | BT.IntOptional
        | BT.Bool
        | BT.Double
        | BT.Tensor
        | BT.ConstTensor
        | BT.TensorAnd
        | BT.TensorList
        | BT.TensorAnd
        | BT.ConstTensor
        | BT.TensorOptions
        | BT.TensorOptionsAnd
        | BT.IntList
        | BT.Device
        | BT.Scalar
        | BT.ScalarType
        | BT.ScalarTypeOptional
        | BT.String -> true
        | _ -> false
    
    if f(arg) then 
        match arg.type_ with
        | BT.Scalar 
        | BT.ScalarOptional when arg.defaultValue.IsSome && not arg.isNullable -> None
        | BT.Scalar when arg.name ="self" -> Some({arg with name = "self_scalar"})
        | BT.TensorOptions
        | BT.TensorOptionsAnd when 
            arg.defaultValue.IsSome && noTensorOptions.Contains(name) ->
                None

        | _ -> Some(arg)
    else 
        if arg.defaultValue.IsSome then None 
        else 
            match arg.type_ with
            | BT.Bool2
            | BT.Bool3
            | BT.Bool4
            | BT.MemoryFormat
            | BT.MemoryFormatOptional
            | BT.Generator
            | BT.Dimname
            | BT.DimnameList
            | BT.DimnameListOptional
            | BT.Storage
            | BT.ConstQuantizerPtr
            | BT.ScalarOptional -> None
            | _ -> failwithf "err %A" arg

module Func = 
    let cTypedArgsList (funcName: string, xs: Arg[]) =
        let xs =  xs |> Array.choose (filterParams funcName)
        [|
            for x in xs do
                let name = match x.name with | "N" -> "n" | _ -> x.name
                match x.type_ with
                | BT.TensorOptions
                | BT.TensorOptionsAnd -> sprintf "int %s_kind, int %s_device" name name
                | BT.Bool2 | BT.Bool3 | BT.Bool4 -> failwithf "todo %A" x
                | BT.IntList -> sprintf "int64_t *%s_data, int %s_len" name name
                //| BT.TensorList when x.isNullable -> sprintf "int %s_kind, int %s_device" name name
                | BT.TensorList (* when not x.isNullable *) -> sprintf "tensor *%s_data, int %s_len" name name
                | BT.DimnameList -> failwithf "todo %A" x
                | BT.String -> Printf.sprintf "char* %s_ptr, int %s_len" name name

                    //| BaseType.Strig -> 
                    //    sprintf "char* %s_ptr, int %s_len" name name
                | _ when x.type_.IsArray -> failwithf "err %A" x
                | _ ->
                    match x.type_ with
                    | BT.Bool -> "int"
                    | BT.Int
                    | BT.IntOptional-> "int64_t"
                    | BT.Double -> "double"
                    | BT.Tensor 
                    | BT.TensorAnd
                    | BT.ConstTensor -> "tensor"
                    | BT.ScalarType 
                    | BT.ScalarTypeOptional -> "int"
                    | BT.Device -> "int"
                    | BT.Scalar -> "scalar"
                    | _ -> (string x.type_) //failwithf "err %A" x
                    |> fun x -> sprintf "%s %s" x name 
        |] |> String.concat ", "

    let cArgsList (funcName: string, args: Arg[]) = 
        let args =  args |> Array.choose (filterParams funcName)
        [|
            for x in args do
                if x.type_.IsArray then
                    match x.type_ with
                    | BT.Bool2 | BT.Bool3 | BT.Bool4 -> failwith "todo"
                    | BT.IntList -> sprintf "torch::IntArrayRef(%s_data, %s_len)" x.name x.name
                    | BT.TensorVector
                    | BT.TensorList ->
                        //if x.isNullable
                        //then sprintf "at::device(device_of_int(%s_device)).dtype(at::ScalarType(%s_kind))" x.name x.name 
                        sprintf "of_carray_tensor(%s_data, %s_len)" x.name x.name
                    | BT.Dimname -> failwith "todo"
                    | _ -> failwithf "err1 %A" x
                else 
                    match x.type_ with
                    | BT.Bool ->  "(bool)" + x.name
                    | BT.Scalar ->  "*" + x.name
                    | Tensor _ ->
                        if x.isNullable
                        then sprintf "(%s ? *%s : torch::Tensor())" x.name x.name  
                        else "*" + x.name 
                    | BT.ScalarTypeOptional 
                    | BT.ScalarType -> sprintf "at::ScalarType(%s)" x.name
                    | BT.Device -> sprintf "device_of_int(%s)" x.name
                    | BT.String ->  sprintf "std::string(%s_ptr, %s_len)"  x.name x.name 
                    | BT.TensorOptionsAnd
                    | BT.TensorOptions -> 
                        sprintf "at::device(device_of_int(%s_device)).dtype(at::ScalarType(%s_kind))" x.name x.name 
                        //sprintf "int %s_kind, int %s_device" x.name x.name
                    | _ -> match x.name with | "N" -> "n" | _ -> x.name
        |] |> String.concat ", "

    let cCall (x: Schema) = 
        match x.methodOfTensor,x.methodOfNamespace with
        | _,Some(true) -> sprintf "torch::%s(%s)" x.name (cArgsList(x.name, x.args))
        | Some(true),_ -> 
            match x.args with
            | [||] -> failwithf "Method calls should have at least one argument %s" x.operatorName
            | _ -> sprintf "%s->%s(%s)" x.args.[0].name x.operatorName (cArgsList(x.name, x.args.[1..]))
        | _,_ -> failwith "err4"

    let selfName = "self"
    let inputName = "input"

    let selfTensor (arg: Arg) = 
        match arg.type_ with
        | Tensor _ -> arg.name = selfName
        | _ -> false

    let inputTensor (arg: Arg) = 
        match arg.type_ with
        | Tensor _ -> arg.name = inputName
        | _ -> false
        
    // Might be Rust specific
//    let typeParameters (t: Schema) = 
//        let needsScalarParameter = 
//            t.args |> Array.exists (fun x -> x.type_ = BT.Scalar || x.type_ = BT.ScalarOptional)
//        let needsTypeParameter = 
//            t.args |> Array.exists (fun x -> x.type_ = BaseType.Tensor && (x.IsArray || x.optional))
//        match needsTypeParameter, needsScalarParameter with
//        | true,true -> "<T: Borrow<Tensor>, S: Into<Scalar>>"
//        | true,false -> "<T: Borrow<Tensor>>"
//        | false,true -> "<S: Into<Scalar>>"
//        | false,false -> ""

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

let funcs = 
    let methods = 
        let baseFunc : Schema = 
            {name = ""; operatorName = ""; overloadName = ""; args = [||]; returns = [||]; depricated = false; methodOfTensor = Some(true); methodOfNamespace = None} 
        let f (name: string)  (baseType: BT) : Arg = 
            { type_ = baseType;  name = name; defaultValue = None; isNullable = false; annotation = None; dynamicType = baseType }
        [|
            "grad", [||]
            "set_requires_grad", [|f "r" BT.Bool|]
            "toType", [|f "scalar_type" BT.ScalarType|]
            "to", [|f "device" BT.Device|]
        |] |> Array.map (fun (name,inputs) -> 
            {baseFunc with name = name; args = [|yield f "self" BT.Tensor; yield! inputs|]})
    let schemas = 
        V2Parser.schemas()
        |> Array.filter (fun x -> not x.depricated) 
        |> Array.filter (fun x -> excludedPrefixes |> Seq.exists (fun y -> x.name.StartsWith(y)) |> not) 
        |> Array.filter (fun x -> excludedSuffixes|> Seq.exists (fun y -> x.name.EndsWith(y)) |> not) 
        |> Array.filter (fun x -> x.overloadName.EndsWith("generator") |> not)
        |> Array.filter (fun x -> x.overloadName.EndsWith("generator_out") |> not)
        |> Array.filter (fun x -> x.overloadName.StartsWith("source_") |> not)
        |> Array.filter (fun x -> excludedFunctions.Contains(x.name) |> not) // 20

    [|yield! methods; yield! schemas|]
    // TODO I'm not sure about this
    //|> Array.filter (function | Function(_) | Method(_) -> true | _ -> false)
//    |> Array.groupBy (fun x -> x.name + match x.overloadName with | "out" -> "_out" | _ -> "")
    |> Array.groupBy (fun x -> x.name)
    //|> Array.groupBy (fun x -> x.firstName)
    |> Array.collect ( function | (x,[|y|]) -> [|(x,y)|] 
                                //| (_,ys) -> ys |> Array.map (fun y -> (sprintf "%s_%s" y.firstName y.secondName, y)))
                                | (_,ys) -> ys |> Array.mapi (fun i y  -> (sprintf "%s%s" y.name (if i = 0 then "" else string i),y)))
    |> Array.map (fun (x,y) -> x.ToLower(),y)
    |> Map.ofArray

let codeGenCpp(funcs: Map<string,Schema>, blackList: Set<string>) = 
    // Create mapping, Rust uses numbers we can use second name
    [|
        yield "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!";
        yield ""
        for KeyValue(exportedName,func) in funcs do
            printfn "%s" exportedName
            if not(blackList.Contains(exportedName)) then
                let cTypedArgsList = Func.cTypedArgsList(func.name,func.args)
                match func.ntensors with
                | Some(Some(ntensors)) -> 
                    yield sprintf "void atg_%s(tensor *out__, %s) {" exportedName cTypedArgsList
                    yield "  PROTECT("
                    yield sprintf "    auto outputs__ = %s;" (Func.cCall func);
                    match ntensors with
                    | 1 -> yield sprintf "    out__[0] = new torch::Tensor(outputs__);"
                    | _ -> 
                        for i = 0 to ntensors - 1 do
                            yield sprintf "    out__[%d] = new torch::Tensor(std::get<%d>(outputs__));" i i 
                    yield! [|"  )"; "}"; ""|]
                | Some(None) -> 
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
                | _ -> () //failwith "err3"
    |]

let codeGenHpp(funcs: Map<string,Schema>) = 
    // Create mapping, Rust uses numbers we can use second name
    [|
        yield "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!";
        yield ""
        for KeyValue(exportedName,func) in funcs do
            let cTypedArgsList = Func.cTypedArgsList(func.name, func.args)
            match func.ntensors with
            | Some(Some(ntensors)) -> 
                yield sprintf "void atg_%s(tensor *, %s);" exportedName cTypedArgsList
            | _ -> 
                yield sprintf  "tensor *atg_%s(%s);" exportedName cTypedArgsList
    |]


let compareWithFilter(filter) = 
    let f (xs: string[]) = 
        match filter with
        | Choice1Of2((skip,take)) ->
            xs |> Array.skip skip |> Array.truncate take
        | Choice2Of2(x) -> xs |> Array.filter (fun y -> y.Contains(x))

    let hpp = 
        funcs |> codeGenHpp  
        |> Array.sort |> f
        |> String.concat "\n"

    let hpp2 = 
        File.ReadAllLines(@"C:\EE\Git\TorchCodeGen\src\Rust\torch_api_generated.h") 
        |> Array.sort |> f
        |> String.concat "\n"

    printf "%s" hpp
    hpp |> DiffPlex.showDiff hpp2 
    

let compareWithFilterCpp(filter) = 
    let f (xs: string[]) = 
        match filter with
        | Choice1Of2((skip,take)) ->
            xs |> Array.skip skip |> Array.truncate take
        | Choice2Of2(x) -> xs |> Array.filter (fun y -> y.Contains(x))

    let hpp = 
        codeGenCpp(funcs,set [])
        |> f
        |> String.concat "\n"

    let hpp2 = 
        File.ReadAllLines(@"C:\EE\Git\TorchCodeGen\src\Rust\torch_api_generated.cpp.h") 
        |> f
        |> String.concat "\n"

    printf "%s" hpp
    hpp |> DiffPlex.showDiff hpp2 

//compareWithFilter(Choice2Of2(""))

compareWithFilterCpp(Choice2Of2(""))

funcs 
|> Map.toArray 
|> Array.map snd
|> Array.tryFind (fun x -> x.name.Contains( "empty_per_channel_affine_quantized"))

funcs 
|> Map.toArray 
|> Array.map snd
|> Array.tryFind (fun x -> x.name.Contains( "index_put_impl_"))
|> Option.get |> fun x -> x.args 

let c = 
    funcs 
    |> Map.toArray 
    |> Array.map snd
    |> Array.tryFind (fun x -> x.name.Contains( "cudnn_init_dropout_state"))
    |> Option.get |> fun x -> x.args |> Array.last 

Func.cArgsList("foo",[|c|])

//let filter1 = Choice1Of2((10,20))
//let filter2 = 

//schemas |> Array.collect (fun x -> x.args) |> Array.filter (fun x -> x.name = "dtype")

//s2 |> Array.filter (fun x -> x.name = "any")

//overloadName = "source_Tensor"
//[|for KeyValue(_,v) in  funcs -> v |]
//|> Array.filter (fun x -> x.name = "clamp")
//|> Array.head |> fun x -> x.args |> Array.last
//
//[|for KeyValue(_,v) in  funcs -> v |]
//|> Array.filter (fun x -> x.name = "addr")
//|> Array.head |> fun x -> x.args |> Array.last


//|> Array.length

//compareWithFilter(Choice2Of2("randint"))

//compareWithFilter(Choice2Of2("set"))

//compareWithFilter(Choice2Of2("clamp"))
//compareWithFilter(Choice2Of2("einsum"))
//compareWithFilter(Choice2Of2("leaky_relu"))
//compareWithFilter(Choice2Of2("any"))
//compareWithFilter(Choice2Of2(""))

//funcs.["randint_out"]
//funcs.["randint"]
//funcs.["randint_like"]
// TODO Some scalars and some are not
// TODO TensorOptions
// TODO Naming doesn't line up

//hpp |> Clippy.copyToClipboard
//hpp2 |> Clippy.copyToClipboard
//
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

