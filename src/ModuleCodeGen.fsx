// NOTE: 229 functions have PythonModuleNN attribute, many do not, e.g. conv2d
// TODO figure out a schema for macros 
#load "CodeGenBase.fsx"
open System
open CodeGenBase
open Clippy
open TorchMetadataParser
open TorchMetadataExtensions
open DiffPlex
open System.IO
open TargetModuleCpp


// 229 schemas have PythonModuleNN
// 45 have simple output and a Tensor input
// 2 have both and these are linear and MKL linear

// 45
let moduleSchemas = 
    schemas 
    |> Array.filter (fun x -> 
        x.outputs = [|simpleTensorOut|] && 
        (x.inputs |> Array.tryHead = Some(simpleTensorInput)) //&& 
        //x.attributes |> Array.exists ((=) PythonModuleNN)
        ) 

let linear = getSchmea("linear","")

[<RequireQualifiedAccess>]
/// Module return types // TODO clean this up later, also consider removing
type X = 
    | Module
    | Tensor
    | Void
    member this.CPP = 
        match this with
        | Module -> "NNModule"
        | Tensor -> "Tensor"
        | Void -> "void"

let genHeader(schema: Func) = 
    let name = schema.firstName |> capitalizeFirst
    let prefix = sprintf "%s" EXPORT_API |> String.indent 4
    let modulePrefix = NNModulePrefix + name
    let standardParams = "const int64_t input_size, const int64_t output_size"
    let tensorInputs = schema.inputs |> Array.filter (fun x -> x.baseType = BaseType.Tensor)
    let extraParams = 
        tensorInputs.[1..] 
        |> Array.choose (fun x -> if x.modifiers.optional then Some(sprintf "const bool with_%s" x.name.Value) else None)

    let padRight = 24
    [|
        yield X.Module, sprintf " %s_ctor(%s, NNAnyModule* outAsAnyModule);" modulePrefix ([|yield standardParams; yield! extraParams|] |> String.concat ", ")
        yield X.Tensor, sprintf " %s_forward(const NNModule module, const Tensor tensor);" modulePrefix 
        for input in tensorInputs.[1..] |> Array.rev do // 1st is known to be "input"
            yield X.Tensor, sprintf " %s_%s(const NNModule module);" modulePrefix input.name.Value
            yield X.Void, sprintf " %s_set_%s(const NNModule module, const Tensor tensor);" modulePrefix input.name.Value
    |] 
    |> Array.map (fun (x,y) -> (sprintf "%s(%s)" prefix  x.CPP).PadRight(padRight) + y)
    |> String.concat Environment.NewLine

let genCpp(schema: Func) = 
    let name = schema.firstName |> capitalizeFirst
    let modulePrefix = NNModulePrefix + name
    let standardParams = "const int64_t input_size, const int64_t output_size"
    let tensorInputs = schema.inputs |> Array.filter (fun x -> x.baseType = BaseType.Tensor)
    let extraParams = 
        tensorInputs.[1..] 
        |> Array.choose (fun x -> if x.modifiers.optional then Some(sprintf "const bool with_%s" x.name.Value) else None)

    let main = 
        func (sprintf "NNModule %s_ctor(%s, NNAnyModule* outAsAnyModule)" modulePrefix ([|yield standardParams; yield! extraParams|] |> String.concat ", ")) (
            macro("CATCH_RETURN_NNModule", true) [|
                yield sprintf "auto opts = torch::nn::%sOptions(input_size, output_size);" name
                for optInput in tensorInputs.[1..] |> Array.filter (fun x -> x.modifiers.optional) do
                    yield sprintf "    opts = opts.%s(with_%s);" optInput.name.Value optInput.name.Value
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
            |])
    
    let fwd = 
        (macro("CATCH_TENSOR",false) [|sprintf "(*module)->as<torch::nn::%s>()->forward(*tensor)" name|])
        |> func(sprintf "Tensor THSNN_%s_forward(const NNModule module, const Tensor tensor)" name) 

    let extraTensors = 
        tensorInputs.[1..] 
        |> Array.rev 
        |> Array.map (fun input -> 
            let iName = input.name.Value
            [|
                func (sprintf "Tensor %s_%s(const NNModule module)" modulePrefix iName) (
                    macro("CATCH_TENSOR",false) [|sprintf "(*module)->as<torch::nn::%s>()->%s" name iName|])
                func (sprintf "void %s_set_%s(const NNModule module, const Tensor %s)" modulePrefix iName iName) (
                    macro("CATCH", true) [| sprintf  "(*module)->as<torch::nn::%s>()->%s = *%s;" name iName iName|])
            |] |> concatWithNewLine)
        |> concatWithNewLine
    [|main; fwd; extraTensors|] 
    |> concatWithNewLine
    |> String.concat Environment.NewLine

open CSharp



let genCSharp(schema: Func) = 
    let name = schema.firstName |> capitalizeFirst
    let modulePrefix = NNModulePrefix + name
    //let standardParams = "const int64_t input_size, const int64_t output_size"
    let tensorInputs = schema.inputs |> Array.filter (fun x -> x.baseType = BaseType.Tensor)
    /// This may include the first input tensor but this should never be optional
    let optionalTensors =  tensorInputs |> Array.filter (fun x -> x.modifiers.optional)
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
                       let iName = input.name.Value
                       let optional = input.modifiers.optional
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
            let ots f = optionalTensors |> Array.map (fun x -> f x.name.Value) |> String.concat ", "
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

linear |> genCSharp |> DiffPlex.showDiff(Linear.csharp)

