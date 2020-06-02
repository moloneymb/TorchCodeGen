#load "V2Parser.fsx"
#load @"DiffPlex.fsx"
#load "Clippy.fsx"
#load "TargetModuleCPP.fsx"
#load "Common.fsx"
open Common
open System
open CodeGenCommon
open V2Parser
open TargetModuleCpp
open Clippy
open DiffPlex
open CSharp
open Cpp
open System
open System.IO


let schemas = V2Parser.schemas()

//let header = "C:\EE\Git\TorchSharp\src\Native\LibTorchSharp\THSNN.h"

let functionNamesTensor = 
    File.ReadAllLines(THSTensorh).[6..] 
    |> Array.choose (fun (line) -> 
        match line.IndexOf("THSTensor_") with
        | -1 -> None
        | n -> 
            // Take until
            let sub = line.Substring(n + 10)
            Some(sub.Substring(0,sub.IndexOf("("))))

let functionNamesNN = //52
//    File.ReadAllLines(THSNNh).[5..] 
//    |> Array.choose (fun (line) -> 
//        match line.IndexOf("THSNN_") with
//        | -1 -> None
//        | n -> 
//            // Take until
//            let sub = line.Substring(n + 6)
//            Some(sub.Substring(0,sub.IndexOf("("))))
  [|"Module_get_parameter"; "Module_get_named_parameters";
    "Module_get_parameters"; "Module_is_training"; "Module_train";
    "Module_eval"; "Module_children_size"; "Module_child"; "Module_name";
    "Module_zero_grad"; "Module_save"; "Module_load"; "Module_register_module";
    "Module_dispose"; "AnyModule_dispose"; "AnyModule_get"; "custom_module";
    "AdaptiveAvgPool2d_ctor"; "AdaptiveAvgPool2d_forward"; "AvgPool2d_ctor";
    "AvgPool2d_forward"; "Conv2d_ctor"; "Conv2d_forward"; "MaxPool2d_ctor";
    "MaxPool2d_forward"; "Dropout_ctor"; "Dropout_forward";
    "FeatureAlphaDropout_ctor"; "FeatureAlphaDropout_forward"; "Linear_ctor";
    "Linear_forward"; "Linear_bias"; "Linear_set_bias"; "Linear_weight";
    "Linear_set_weight"; "ReLU_ctor"; "ReLU_forward"; "Sequential_ctor";
    "Sequential_push_back"; "Sequential_forward"; "Optimizer_zeroGrad";
    "Optimizer_getParameters"; "Optimizer_step"; "Optimizer_dispose";
    "binary_cross_entropy"; "mse_loss"; "nll_loss"; "poisson_loss";
    "Adam_ctor"; "SGD_ctor"; "initUniform"; "initKaimingUniform"|]

// Matching done by hand
let foundSchemasNN = 
    [|
        "adaptive_avg_pool2d"; "avg_pool2d"; "conv2d"; 
        "max_pool2d"; "dropout"; "feature_alpha_dropout";
        "linear"; "relu"; "binary_cross_entropy"; "mse_loss"; 
        "nll_loss"; "poisson_nll_loss"
    |]


//schemas |> Array.filter (fun x -> x.name.Contains("linear")) |> Array.iter printSchemaSimple
//let linear = schemas |> Array.find (fun x -> x.name = "linear" && x.overloadName = "") 

[<RequireQualifiedAccess>]
/// Module return types // TODO clean this up later, also consider removing
type ReturnType = 
    | Module
    | Tensor
    | Void
    member this.CPP = 
        match this with
        | Module -> "NNModule"
        | Tensor -> "Tensor"
        | Void -> "void"

type RT = ReturnType


let argName(x:string) = 
    match x with
    | "N" -> "n"
    | "output_size" -> "sizes"
    | _ -> x

let argCSignature(x:Arg) = 
    let name = argName x.name
    match x.type_ with
    | BT.TensorOptions
    | BT.TensorOptionsAnd -> sprintf "int %s_kind, int %s_device" name name
    | BT.Bool2 | BT.Bool3 | BT.Bool4 -> failwithf "todo %A" x
    | BT.IntList -> sprintf "const int64_t* %s, const int %s_len" name name
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
    |> fun y -> if x.isNullable then sprintf "const bool with_%s, %s" x.name y else y



let genHeader(schema: Schema) = 
    let name = schema.name |> underscoreToCamel |> capitalizeFirst
    let prefix = sprintf "%s" EXPORT_API |> String.indent 4
    let modulePrefix = NNModulePrefix + name
    //let standardParams = "const int64_t input_size, const int64_t output_size"
    let tensorInputs = schema.args |> Array.filter (fun x -> match x.type_ with | Tensor _ -> true | _ -> false)
    let extraParams = 
        schema.args.[1..] 
        //|> Array.filter (fun x -> FilterDefault && x.defaultValue.IsNone)
        |> Array.map argCSignature
        //|> Array.choose (fun x -> 
        //    if x.isNullable then Some(sprintf "const bool with_%s" x.name) 
        //    else 
        //        Some(sprintf "%s %s" (string x.type_) x.name))

    let padRight = 24
    [|
        yield RT.Module, sprintf " %s_ctor(%s, NNAnyModule* outAsAnyModule);" modulePrefix ([|(*yield standardParams;*) yield! extraParams|] |> String.concat ", ")
        yield RT.Tensor, sprintf " %s_forward(const NNModule module, const Tensor tensor);" modulePrefix 
        for input in tensorInputs.[1..] |> Array.rev do // 1st is known to be "input"
            yield RT.Tensor, sprintf " %s_%s(const NNModule module);" modulePrefix input.name
            yield RT.Void, sprintf " %s_set_%s(const NNModule module, const Tensor tensor);" modulePrefix input.name
    |] 
    |> Array.map (fun (x,y) -> (sprintf "%s(%s)" prefix  x.CPP).PadRight(padRight) + y)
    //|> String.concat Environment.NewLine


//schemas |> Array.find (fun y ->  y.name.Contains("adaptive"))

let i = 2

let f  = 
    let y = foundSchemasNN.[i] |> underscoreToCamel |> capitalizeFirst
    fun (x:string) -> x.Contains(y)


foundSchemasNN
|> Array.map (fun x -> printfn "%s" x; schemas |> Array.find (fun y ->  y.name = x))
|> Array.collect genHeader
|> Array.filter f
|> Array.sort
|> String.concat System.Environment.NewLine
|> DiffPlex.showDiff (
    File.ReadAllLines(THSNNh) 
    |> Array.filter f
    |> Array.sort 
    |> String.concat System.Environment.NewLine)

DiffPlex.diffViewer.Value.IgnoreWhiteSpace <- true
DiffPlex.diffViewer.Value.IgnoreCase <- true
DiffPlex.diffViewer.Value.ShowSideBySide()

schemas |> Array.filter (fun x -> x.name = foundSchemasNN.[i]) |> Array.length

File.ReadAllLines(@"C:\EE\Git\TorchSharp\src\Native\LibTorchSharp\THSTensor.h")
|> Array.filter (fun x -> not((String.IsNullOrWhiteSpace(x)) || x.StartsWith("//")))
|> fun xs -> File.WriteAllLines(@"C:\EE\Git\TorchSharp\src\Native\LibTorchSharp\THSTensor.h",xs)

//let genCpp(schema: Schema) = 
//    let name = schema.name |> underscoreToCamel |> capitalizeFirst
//    let modulePrefix = NNModulePrefix + name
//    let standardParams = "const int64_t input_size, const int64_t output_size"
////    let tensorInputs = schema.args |> Array.filter (fun x -> match x.type_ with | Tensor _ -> true | _ -> false)
////    let extraParams = 
////        tensorInputs.[1..] 
////        |> Array.choose (fun x -> if x.isNullable then Some(sprintf "const bool with_%s" x.name) else None)
//
//    let main = 
//        func (sprintf "NNModule %s_ctor(%s, NNAnyModule* outAsAnyModule)" modulePrefix ([|(*yield standardParams;*) (* yield! extraParams *)|] |> String.concat ", ")) (
//            macro("CATCH_RETURN_NNModule", true) [|
//                yield sprintf "auto opts = torch::nn::%sOptions(input_size, output_size);" name
//                failwith "todo"
////                for optInput in tensorInputs.[1..] |> Array.filter (fun x -> x.isNullable) do
////                    yield sprintf "opts = opts.%s(with_%s);" optInput.name optInput.name
//                yield ""
//                yield sprintf "auto mod = std::make_shared<torch::nn::%sImpl>(opts);" name
//                yield ""
//                yield "// Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means"
//                yield "// a Module can only be boxed to AnyModule at the point its static type is known)."
//                yield! 
//                    ifThen("outAsAnyModule != NULL") [|
//                        yield sprintf "auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::%sImpl>(*mod));" name
//                        yield sprintf "*outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);"
//                    |]
//                yield "res = new std::shared_ptr<torch::nn::Module>(mod);"
//            |]
//            |> semicolon)
//    
//    let fwd = 
//        (macro("CATCH_TENSOR",false) [|sprintf "(*module)->as<torch::nn::%s>()->forward(*tensor)" name|])
//        |> semicolon
//        |> func(sprintf "Tensor THSNN_%s_forward(const NNModule module, const Tensor tensor)" name) 
//
//    let extraTensors = 
//        tensorInputs.[1..] 
//        |> Array.rev 
//        |> Array.map (fun input -> 
//            let iName = input.name
//            [|
//                func (sprintf "Tensor %s_%s(const NNModule module)" modulePrefix iName) (
//                    macro("CATCH_TENSOR",false) [|sprintf "(*module)->as<torch::nn::%s>()->%s" name iName|]
//                    |> semicolon) 
//                func (sprintf "void %s_set_%s(const NNModule module, const Tensor %s)" modulePrefix iName iName) (
//                    macro("CATCH", true) [| sprintf  "(*module)->as<torch::nn::%s>()->%s = *%s;" name iName iName|]
//                    )
//            |] |> concatWithNewLine)
//        |> concatWithNewLine
//    [|main; fwd; extraTensors|] 
//    |> concatWithNewLine
//    |> indent
//    |> Array.map (fun x -> if String.IsNullOrWhiteSpace(x) then "" else x)
//    //|> String.concat Environment.NewLine
//
//let genCSharp(schema: Schema) = 
//    let name = schema.name |> underscoreToCamel |> capitalizeFirst
//    let modulePrefix = NNModulePrefix + name
//    //let standardParams = "const int64_t input_size, const int64_t output_size"
//    let tensorInputs = schema.args |> Array.filter (fun x -> match x.type_ with | Tensor _ -> true | _ -> false)
//    /// This may include the first input tensor but this should never be optional
//    let optionalTensors =  tensorInputs |> Array.filter (fun x -> x.isNullable)
//    let head  =
//        [|
//             """// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information."""
//             "using System;"
//             "using System.Collections.Generic;"
//             "using System.Diagnostics;"
//             "using System.Runtime.InteropServices;"
//             "using TorchSharp.Tensor;"
//        |]
//    let body = 
//        namespace_ "TorchSharp.NN" [|
//            yield! 
//                func (sprintf "public class %s : Module" name) ([|
//                   yield sprintf "internal %s (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }" name
//                   yield ""
//                   yield! func(sprintf "public new static %s Load (String modelPath)" name) ([|
//                            "var res = Module.Load (modelPath);"
//                            "Torch.CheckForErrors ();"
//                            sprintf "return new %s (res.handle.DangerousGetHandle(), IntPtr.Zero);" name
//                   |])
//                   yield ""
//                   yield! extern_ (sprintf "IntPtr THSNN_%s_forward (Module.HType module, IntPtr tensor);" name)
//                   yield ""
//                   yield! func("public TorchTensor Forward (TorchTensor tensor)") ([|
//                        sprintf "var res = THSNN_%s_forward (handle, tensor.Handle);" name
//                        "Torch.CheckForErrors ();"
//                        "return new TorchTensor (res);"
//                       |])
//                   for input in tensorInputs.[1..] |> Array.rev do
//                       let iName = input.name
//                       let optional = input.isNullable
//                       yield! extern_ (sprintf "IntPtr THSNN_%s_%s (Module.HType module);" name iName)
//                       yield! extern_ (sprintf "void THSNN_%s_set_%s (Module.HType module, IntPtr tensor);" name iName)
//                       yield ""
//                       yield! getSetMember (sprintf "public TorchTensor%s %s" (if optional then "?" else "") (iName |> capitalizeFirst), 
//                                [|
//                                    sprintf "var res = THSNN_%s_%s (handle);" name iName; 
//                                    "Torch.CheckForErrors ();"; 
//                                    if optional then "return ((res == IntPtr.Zero) ? null : (TorchTensor ?)new TorchTensor (res));"
//                                    else "return new TorchTensor (res);" |], 
//                                [|
//                                    sprintf "THSNN_%s_set_%s (handle, %s);" name iName 
//                                        (if optional then "(value.HasValue ? value.Value.Handle : IntPtr.Zero)" else "value.Handle")
//                                    "Torch.CheckForErrors ();" |])
//        // NOTE: Defaults may not always need to be true...
//                |])
//            let ots f = optionalTensors |> Array.map (fun x -> f x.name) |> String.concat ", "
//            /// start name with a capital letter
//            let otsWithCap f = ots (fun x -> x |> capitalizeFirst |> f)
//            yield! func("public static partial class Modules") ([|
//                yield! extern_ (sprintf "IntPtr THSNN_%s_ctor (long input_size, long output_size, %sout IntPtr pBoxedModule);" name (ots (sprintf "bool with_%s") |> appendParam))
//                yield ""
//                // TODO figure out defaults
//                yield! func(sprintf "static public %s %s (long inputSize, long outputSize%s)" name name (otsWithCap (sprintf "bool has%s = true") |> prependParam)) ([|
//                    sprintf "var res = THSNN_%s_ctor (inputSize, outputSize, %sout var boxedHandle);" name (otsWithCap (sprintf "has%s") |> appendParam)
//                    "Torch.CheckForErrors ();"
//                    sprintf "return new %s (res, boxedHandle);" name
//                    |])
//                |])
//            yield! 
//                [|"return d.Forward (x);"|]
//                |> using ("var d = Modules.Linear (inputSize, outputSize, hasBias)") 
//                |> func ("static public TorchTensor Linear (TorchTensor x, long inputSize, long outputSize, bool hasBias = true)") 
//                |> func("public static partial class Functions")
//        |] 
//        |> Array.map (fun x -> x.TrimEnd())
//        //|> String.concat Environment.NewLine
//    [|head;[|"";""|];body|] |> Array.collect id


//File.ReadAllLines(THSNNh).Length

//let headerFile = 

//foundSchemas 
//|> Array.map (fun x -> printfn "%s" x; schemas |> Array.find (fun y ->  y.name = x))

// TODO convert to PascalCase
//(schemas |> Array.filter (fun y ->  y.name = "dropout")) |> Array.map (fun x -> (x.name, x.operatorName, x.overloadName))



//def pascal_to_underscore(name):
//    underscore_name = re.sub(r'(?!^)([A-Z])([a-z])', r'_\1\2', name).lower()
//    return underscore_name + "_" if underscore_name in fsharp_keywords else underscore_name

//schemas |> Array.filter (fun x -> x.name.Contains("poisson")) |> Array.map (fun x -> x.name)

//genHeader()
//File.ReadAllText(THSNNh)

//foundSchemas 
//schemas |> Array.filter (fun x -> x.name.Contains("linear")) |> Array.iter printSchemaSimple
//let linear = schemas |> Array.find (fun x -> x.name = "linear" && x.overloadName = "") 

//linear |> genHeader |> DiffPlex.showDiff(Linear.hearder)
//linear |> genCpp |> DiffPlex.showDiff(Linear.cpp)
//linear |> genCSharp |> DiffPlex.showDiff(Linear.csharp)


// CSharp Code Generation

(*
let adaptiveAvgPool2D = "AdaptiveAvgPool2D.cs"
let avgPool2D = "AvgPool2D.cs"
let conv2D = "Conv2D.cs"
let dropout = "Dropout.cs"
let linear = "Linear.cs"
let logSoftMax = "LogSoftMax.cs"
let featureDropout = "FeatureDropout.cs"
let maxPool2D = "MaxPool2D.cs"
let reLu = "ReLu.cs"
*)
