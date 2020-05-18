#load "CodeGenBase.fsx"

open Clippy
open TorchMetadataParser
open TorchMetadataExtensions
open DiffPlex
open System.IO
open TargetModuleCpp
open CodeGenBase

let getCppReturn(schema: Func) = 
    if schema.outputs = [|simpleTensorOut|] then
        "Tensor"
    else failwith "todo"

let getCppParams(schema: Func) = 
    [|
        for input in schema.inputs do
            match input.baseType with
            | BaseType.Scalar when input.modifiers = emptyModifiers -> 
                yield sprintf "const Scalar %s" input.name.Value
            | BaseType.ScalarType when input.name = Some("dtype") -> 
                yield sprintf "const int8_t dtype" 
            //| BaseType.Device -> yield "const char * device"
            | BaseType.Astrix -> () // TODO
            | BaseType.Layout -> () // TODO
            | BaseType.Device -> 
                yield "const char * device"
            | BaseType.Int when input.modifiers.array.IsSome ->
                yield sprintf "const int64_t * %s" input.name.Value
                yield sprintf "const int %s_length" input.name.Value
            | BaseType.Bool when input.name = Some("pin_memory") -> ()
            | _ -> failwithf "todo support CppParams %A" input
        yield "const bool requires_grad" // TODO figure out if/why we need this
    |] 


let getCppOptions(schema: Func) = 
    [|
        for input in schema.inputs do
            match input.baseType with
            | BaseType.ScalarType when input.name = Some("dtype") ->
                yield ".dtype(at::ScalarType(dtype))"
            | BaseType.Astrix -> () // TODO
            | BaseType.Layout -> () // TODO
            | BaseType.Device -> 
                yield ".device(device)"
            | BaseType.Bool -> ()
            | BaseType.Scalar -> ()
            | BaseType.Int -> ()
            | _ -> failwithf "todo support CppOptions %A" input
        yield ".requires_grad(requires_grad)" // TODO figure out if/why we need this
    |] 

// TODO find out what requires options

let getCppTorchParams(schema: Func) = 
    [|
        for input in schema.inputs do
            match input.baseType with
            | BaseType.Int when input.modifiers.array.IsSome ->
                yield sprintf "at::IntList(%s,%s_length)" input.name.Value input.name.Value
            | BaseType.Scalar -> yield "*" + input.name.Value
            | _ -> ()
        yield "options"
    |]

let getCppFunction (name:string) (schema: Func)  = 
    [|
        yield sprintf "%s THSTensor_%s(" (getCppReturn(schema)) name //schema.firstName
        yield getCppParams schema 
            |> Array.map (fun x -> "    " + x) 
            |> String.concat ",\r\n" 
            |> sprintf "%s)\r\n{"
        // TODO check if we need options
        yield "    auto options = at::TensorOptions()"
        yield getCppOptions(schema)
            |> Array.map (fun x -> "        " + x) 
            |> String.concat "\r\n" 
            |> sprintf "%s;\r\n"

        let torchParams = schema |> getCppTorchParams |> String.concat ", "
        if schema.outputs = [|simpleTensorOut|] then
            yield sprintf "    CATCH_RETURN_TENSOR(torch::%s(%s));" name (* schema.firstName *)  torchParams
        else failwith "err"

        yield "}"
    |] |> String.concat "\r\n"

/// Generating
//let genearting = 
//    [|
//        "arange"
//        "zeros"
//    |]
//
///// This
//let nameMapping = 
//    [|
//        "arange", ("arange","start_step")
//        "zeros", ("","zeros")
//        "ones", ("","ones")
//    |] |> Map.ofArray
//getSchmea(nameMapping.["arange"]) |> getCppFunction "arange" |> copyToClipboard
//getSchmea(nameMapping.["zeros"]) |> getCppFunction "zeros" |> copyToClipboard
//getSchmea(nameMapping.["ones"]) |> getCppFunction "ones" |> copyToClipboard
//getSchmea("","empty_strided") |> getCppFunction "empty" |> copyToClipboard
//getSchmea("","rand") |> getCppFunction "rand" 
//getSchmea("","ones") |> getCppFunction "ones"
//searchSchemas("rand")  |> Array.map getSchmea  |> Array.filter (fun x -> x.secondName = "rand" ) |> Array.iter printSchemaSimple
//getSchmea("empty","out") |> printSchemaSimple
//getSchmea("","empty_strided") |> printSchemaSimple

let makeScalar(name,cppType : ScalarType) = 
    sprintf """Tensor THSTensor_new%sScalar(%s data, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::%s))
        .requires_grad(requires_grad);
    CATCH_RETURN_TENSOR(torch::tensor(data, options));
}""" name (cppType |> scalarTypeToCppType) (cppType |> scalarTypeToString)

//makeScalar(scalars.[0]) //|> copyToClipboard

type Func with
    member this.TensorParams = 
        this.inputs |> Array.filter (fun x -> x.baseType = BaseType.Tensor)

// TODO unary functions 575
let unaryTensorFunctions = 
    schemas |> Array.filter (fun x -> x.TensorParams.Length = 1)

// TODO binary functions 361
let binaryTensorFunctions = 
    schemas |> Array.filter (fun x -> x.TensorParams.Length = 2)

//unaryTensorFunctions |> Array.map (fun x -> x.TensorParams.[0].name) |> Array.countBy id

binaryTensorFunctions.Length
unaryTensorFunctions.Length

unaryTensorFunctions |> Seq.tryFind (fun x -> x.firstName = "" && x.secondName = "cos")

// 125 fit this simple mold
let unaryToCpp (name: string) (schema: Func) : Result<string,string> = 
    match schema.inputs, schema.outputs with
    | [|input|], [|output|] ->
        if (input.baseType = BaseType.Tensor && output.baseType = BaseType.Tensor) &&
           ((input.modifiers = emptyModifiers && output.modifiers = emptyModifiers) || 
            (input.modifiers = alphaBangModifiers && output.modifiers = alphaBangModifiers)) then
            Ok(sprintf """Tensor THSTensor_%s(const Tensor %s)
{
    CATCH_RETURN_TENSOR(%s->%s());
}"""  name input.name.Value input.name.Value name )
        else
            Error(sprintf "%s input/output modifiers not supported %A, %A" 
                    name input.modifiers output.modifiers)
    | _ -> Error(sprintf "%s non - standard unary" name)

// 125 fit this simple mold
//unaryTensorFunctions |> Array.choose (fun x -> match unaryToCpp x.secondName x with | Ok(y) -> Some(x.firstName, x.secondName) | _ -> None) |> Array.length
//
//unaryTensorFunctions |> Array.find (fun x -> x.firstName = "add" && x.secondName = "Scalar")

// 52
//unaryTensorFunctions |> Array.filter (fun x ->  x.secondName = "Scalar") |> Array.length

// 52
//binaryTensorFunctions |> Array.filter (fun x -> x.secondName = "Tensor") |> Seq.head
// Astrix??, and alpha

//tensorBinaryTensorFunctions |> Array.filter (fun x -> x.secondName = "Tensor") 

//let binaryToCpp (name: string) (schema: Func) : Result<string,string> = 
//    match schema.inputs, schema.outputs with
//    | [|inputA;inputB|], [|output|] ->
//        if (inputA.baseType = BaseType.Tensor && inputB.baseType = BaseType.Tensor && output.baseType = BaseType.Tensor) &&
//           ((inputA.modifiers = emptyModifiers && output.modifiers = emptyModifiers) || 
//            (inputA.modifiers = alphaBangModifiers && output.modifiers = alphaBangModifiers)) then
//            Ok(sprintf """Tensor THSTensor_%s(const Tensor %s)
//{
//    CATCH_RETURN_TENSOR(%s->%s());
//}"""  name input.name.Value input.name.Value name )
//        else
//            Error(sprintf "%s input/output modifiers not supported %A, %A" 
//                    name input.modifiers output.modifiers)
//    | _ -> Error(sprintf "%s non - standard unary" name)


//Tensor THSTensor_add(const Tensor left, const Tensor right, const Scalar alpha)
//{
//    CATCH_RETURN_TENSOR(left->add(*right, *alpha));
//}
//
//Tensor THSTensor_add_(const Tensor left, const Tensor right, const Scalar alpha)
//{
//    CATCH_RETURN_TENSOR(left->add_(*right, *alpha));
//}




