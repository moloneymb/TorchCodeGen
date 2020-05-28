#load "V2Parser.fsx"
#load @"DiffPlex.fsx"
#load "Clippy.fsx"
#load "TargetModuleCPP.fsx"
open System
open CodeGenCommon
open V2Parser
open TargetModuleCpp
open Clippy
open DiffPlex
open CSharp
open Cpp

let schemas = V2Parser.schemas()

schemas |> Array.filter (fun x -> x.name.Contains("linear")) |> Array.iter printSchemaSimple
let linear = schemas |> Array.find (fun x -> x.name = "linear" && x.overloadName = "") 

let [<Literal>] NNModulePrefix = "THSNN_"
let [<Literal>] EXPORT_API = "EXPORT_API"

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

let genHeader(schema: Schema) = 
    let name = schema.name  |> capitalizeFirst
    let prefix = sprintf "%s" EXPORT_API |> String.indent 4
    let modulePrefix = NNModulePrefix + name
    let standardParams = "const int64_t input_size, const int64_t output_size"
    let tensorInputs = schema.args |> Array.filter (fun x -> match x.type_ with | Tensor _ -> true | _ -> false)
    let extraParams = 
        tensorInputs.[1..] 
        |> Array.choose (fun x -> if x.isNullable then Some(sprintf "const bool with_%s" x.name) else None)

    let padRight = 24
    [|
        yield RT.Module, sprintf " %s_ctor(%s, NNAnyModule* outAsAnyModule);" modulePrefix ([|yield standardParams; yield! extraParams|] |> String.concat ", ")
        yield RT.Tensor, sprintf " %s_forward(const NNModule module, const Tensor tensor);" modulePrefix 
        for input in tensorInputs.[1..] |> Array.rev do // 1st is known to be "input"
            yield RT.Tensor, sprintf " %s_%s(const NNModule module);" modulePrefix input.name
            yield RT.Void, sprintf " %s_set_%s(const NNModule module, const Tensor tensor);" modulePrefix input.name
    |] 
    |> Array.map (fun (x,y) -> (sprintf "%s(%s)" prefix  x.CPP).PadRight(padRight) + y)
    |> String.concat Environment.NewLine


let genCpp(schema: Schema) = 
    let name = schema.name |> capitalizeFirst
    let modulePrefix = NNModulePrefix + name
    let standardParams = "const int64_t input_size, const int64_t output_size"
    let tensorInputs = schema.args |> Array.filter (fun x -> match x.type_ with | Tensor _ -> true | _ -> false)
    let extraParams = 
        tensorInputs.[1..] 
        |> Array.choose (fun x -> if x.isNullable then Some(sprintf "const bool with_%s" x.name) else None)

    let main = 
        func (sprintf "NNModule %s_ctor(%s, NNAnyModule* outAsAnyModule)" modulePrefix ([|yield standardParams; yield! extraParams|] |> String.concat ", ")) (
            macro("CATCH_RETURN_NNModule", true) [|
                yield sprintf "auto opts = torch::nn::%sOptions(input_size, output_size);" name
                for optInput in tensorInputs.[1..] |> Array.filter (fun x -> x.isNullable) do
                    yield sprintf "opts = opts.%s(with_%s);" optInput.name optInput.name
                yield ""
                yield sprintf "auto mod = std::make_shared<torch::nn::%sImpl>(opts);" name
                yield ""
                yield "// Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means"
                yield "// a Module can only be boxed to AnyModule at the point its static type is known)."
                yield! 
                    ifThen("outAsAnyModule != NULL") [|
                        yield sprintf "auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::%sImpl>(*mod));" name
                        yield sprintf "*outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);"
                    |]
                yield "res = new std::shared_ptr<torch::nn::Module>(mod);"
            |]
            |> semicolon)
    
    let fwd = 
        (macro("CATCH_TENSOR",false) [|sprintf "(*module)->as<torch::nn::%s>()->forward(*tensor)" name|])
        |> semicolon
        |> func(sprintf "Tensor THSNN_%s_forward(const NNModule module, const Tensor tensor)" name) 

    let extraTensors = 
        tensorInputs.[1..] 
        |> Array.rev 
        |> Array.map (fun input -> 
            let iName = input.name
            [|
                func (sprintf "Tensor %s_%s(const NNModule module)" modulePrefix iName) (
                    macro("CATCH_TENSOR",false) [|sprintf "(*module)->as<torch::nn::%s>()->%s" name iName|]
                    |> semicolon) 
                func (sprintf "void %s_set_%s(const NNModule module, const Tensor %s)" modulePrefix iName iName) (
                    macro("CATCH", true) [| sprintf  "(*module)->as<torch::nn::%s>()->%s = *%s;" name iName iName|]
                    )
            |] |> concatWithNewLine)
        |> concatWithNewLine
    [|main; fwd; extraTensors|] 
    |> concatWithNewLine
    |> indent
    |> Array.map (fun x -> if String.IsNullOrWhiteSpace(x) then "" else x)
    |> String.concat Environment.NewLine

let genCSharp(schema: Schema) = 
    let name = schema.name |> capitalizeFirst
    let modulePrefix = NNModulePrefix + name
    //let standardParams = "const int64_t input_size, const int64_t output_size"
    let tensorInputs = schema.args |> Array.filter (fun x -> match x.type_ with | Tensor _ -> true | _ -> false)
    /// This may include the first input tensor but this should never be optional
    let optionalTensors =  tensorInputs |> Array.filter (fun x -> x.isNullable)
    let head  = """// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;"""
    let body = 
        namespace_ "TorchSharp.NN" [|
            yield! 
                func (sprintf "public class %s : Module" name) ([|
                   yield sprintf "internal %s (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }" name
                   yield ""
                   yield! func(sprintf "public new static %s Load (String modelPath)" name) ([|
                            "var res = Module.Load (modelPath);"
                            "Torch.CheckForErrors ();"
                            sprintf "return new %s (res.handle.DangerousGetHandle(), IntPtr.Zero);" name
                   |])
                   yield ""
                   yield! extern_ (sprintf "IntPtr THSNN_%s_forward (Module.HType module, IntPtr tensor);" name)
                   yield ""
                   yield! func("public TorchTensor Forward (TorchTensor tensor)") ([|
                        sprintf "var res = THSNN_%s_forward (handle, tensor.Handle);" name
                        "Torch.CheckForErrors ();"
                        "return new TorchTensor (res);"
                       |])
                   for input in tensorInputs.[1..] |> Array.rev do
                       let iName = input.name
                       let optional = input.isNullable
                       yield! extern_ (sprintf "IntPtr THSNN_%s_%s (Module.HType module);" name iName)
                       yield! extern_ (sprintf "void THSNN_%s_set_%s (Module.HType module, IntPtr tensor);" name iName)
                       yield ""
                       yield! getSetMember (sprintf "public TorchTensor%s %s" (if optional then "?" else "") (iName |> capitalizeFirst), 
                                [|
                                    sprintf "var res = THSNN_%s_%s (handle);" name iName; 
                                    "Torch.CheckForErrors ();"; 
                                    if optional then "return ((res == IntPtr.Zero) ? null : (TorchTensor ?)new TorchTensor (res));"
                                    else "return new TorchTensor (res);" |], 
                                [|
                                    sprintf "THSNN_%s_set_%s (handle, %s);" name iName 
                                        (if optional then "(value.HasValue ? value.Value.Handle : IntPtr.Zero)" else "value.Handle")
                                    "Torch.CheckForErrors ();" |])
        // NOTE: Defaults may not always need to be true...
                |])
            let ots f = optionalTensors |> Array.map (fun x -> f x.name) |> String.concat ", "
            /// start name with a capital letter
            let otsWithCap f = ots (fun x -> x |> capitalizeFirst |> f)
            yield! func("public static partial class Modules") ([|
                yield! extern_ (sprintf "IntPtr THSNN_%s_ctor (long input_size, long output_size, %sout IntPtr pBoxedModule);" name (ots (sprintf "bool with_%s") |> appendParam))
                yield ""
                // TODO figure out defaults
                yield! func(sprintf "static public %s %s (long inputSize, long outputSize%s)" name name (otsWithCap (sprintf "bool has%s = true") |> prependParam)) ([|
                    sprintf "var res = THSNN_%s_ctor (inputSize, outputSize, %sout var boxedHandle);" name (otsWithCap (sprintf "has%s") |> appendParam)
                    "Torch.CheckForErrors ();"
                    sprintf "return new %s (res, boxedHandle);" name
                    |])
                |])
            yield! 
                [|"return d.Forward (x);"|]
                |> using ("var d = Modules.Linear (inputSize, outputSize, hasBias)") 
                |> func ("static public TorchTensor Linear (TorchTensor x, long inputSize, long outputSize, bool hasBias = true)") 
                |> func("public static partial class Functions")
        |] 
        |> Array.map (fun x -> x.TrimEnd())
        |> String.concat Environment.NewLine
    head + Environment.NewLine + Environment.NewLine + body


//linear |> genHeader |> DiffPlex.showDiff(Linear.hearder)
//linear |> genCpp |> DiffPlex.showDiff(Linear.cpp)
//linear |> genCSharp |> DiffPlex.showDiff(Linear.csharp)



