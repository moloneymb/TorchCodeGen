#load "V2CodeGenLLTorchSharp.fsx"
// TODO - figure out 64bit

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
open System.Runtime.InteropServices

let aditionalFiltered = 
    set [
        // These have mixed return tuples
        "_batch_norm_impl_index","" // (Tensor * Tensor * Tensor * Tensor * int)
        "fbgemm_linear_quantize_weight","" // (Tensor * Tensor * double * int)
        "convolution_backward_overrideable","" // Bool3
        "from_file","" // BoolOptional
        "to","device" // MemoryFormatOptional
        "item","" //Scalar return type
        "_local_scalar_dense","" // Scalar return type
        "set_quantizer_", "" // ConstQuantizerPtr
        "qscheme", "" // QScheme
    ]

let extraFilteredSchemas() = 
    V2Filtered.filtered_schemas()
    |> Array.filter (fun x -> x.args |> Array.exists (fun x -> match x.type_ with | BT.Dimname | BT.DimnameList | BT.DimnameListOptional -> true | _ -> false) |> not)
    |> Array.filter (fun x -> x.name.EndsWith("backward") |> not)
    |> Array.filter (fun x -> x.name.EndsWith("backward_out") |> not)
    |> Array.filter (fun x -> x.args |> Array.exists (fun x -> match x.type_ with | BT.Generator -> true | _ -> false) |> not)
    |> Array.filter (fun x -> aditionalFiltered.Contains(x.name,x.overloadName) |> not)
    |> Array.filter (fun x -> x.args |> Array.exists (fun x -> match x.type_ with | BT.MemoryFormat | BT.MemoryFormatOptional -> true | _ -> false) |> not)
    |> Array.choose (fun schema -> 
            try
                let x = schema |> V2CodeGenLLTorchSharp.genCpp
                let y = schema |> V2CodeGenLLTorchSharp.genHeader (true,true)
                let z = schema |> V2CodeGenLLTorchSharp.genCSharp
                Some(schema)
            with
            | _ -> None
    )

// Extra Extra filtered
let schemas = extraFilteredSchemas()

let CSharpTarget = @"C:\EE\Git\TorchSharp2\src\TorchSharp\Tensor\TorchTensorG.cs"
let CPPTarget = @"C:\EE\Git\TorchSharp2\src\Native\LibTorchSharp\THSTensorG.cpp"
let HTarget = @"C:\EE\Git\TorchSharp2\src\Native\LibTorchSharp\THSTensorG.h"

let HeaderHeader = 
    [|
        "#pragma once"
        "#include \"../Stdafx.h\""
        "#include \"TH/THTensor.h\""
        "#include \"torch/torch.h\""
        "#include \"Utils.h\""
        "EXPORT_API(long) THSTensor_code_gen_timestamp();"    
    |]

let codeGenTime = DateTime.Now.Ticks

let CPPHeader = 
    [|
        "#include \"THSTensorG.h\""
        "#include <iostream>"
        "#include <fstream>"
        sprintf """ long THSTensor_code_gen_timestamp() { return (long) %i; }""" codeGenTime
    |]

let CSharpHeader = 
    [|
        "using System;"
        "using System.Linq;"
        "using System.Runtime.CompilerServices;"
        "using System.Runtime.InteropServices;"
        "using System.Text;"
    |]

let DLLImport = "[DllImport(\"LibTorchSharp\")]"

let CSharpBody = 
    [| 
        yield! 
            [|
                DLLImport
                "extern static long THSTensor_code_gen_timestamp();"
                "public static long CodeGenTimestamp { get { return THSTensor_code_gen_timestamp(); } }"
            |]
        for schema in schemas do
            yield DLLImport
            yield schema |> V2CodeGenLLTorchSharp.genImport(true)
            yield ""
            yield! schema |> V2CodeGenLLTorchSharp.genCSharp

    |] 
    |> func ("public partial class TorchTensor : IDisposable")
    |> func ("namespace TorchSharp.Tensor")


// TODO - underscore for name before overload..
// TODO - filter out if any contains MemoryFormat


File.WriteAllLines(CSharpTarget, [|yield! CSharpHeader; yield! CSharpBody|])
File.WriteAllLines(CPPTarget, [|yield! CPPHeader; for schema in schemas do yield! (V2CodeGenLLTorchSharp.genCpp  schema)|])
File.WriteAllLines(HTarget, [|yield! HeaderHeader; for schema in schemas do yield! (V2CodeGenLLTorchSharp.genHeader (true,true) schema |> addFinalSemiColon)|])

//File.WriteAllLines(HTarget, [|yield! HeaderHeader; yield!  |])

open System.Diagnostics

// Doesn't work, msbuild issue, no time to track it down now

//let runProcess (workingDirectory : string option) (name:string) (envs:(string*string)[]) (args:string)  =
//        printfn "Running: %s %s" name args
//        let psi = ProcessStartInfo()
//        psi.FileName <- name
//        psi.Arguments <- args
//        psi.ErrorDialog <- true
//        workingDirectory |> Option.iter (fun x -> psi.WorkingDirectory <- x)
//        psi.UseShellExecute <- false
//        let p = new Process()
//        p.StartInfo <- psi
//        p.ErrorDataReceived.Add(fun e -> printfn "Error %s" e.Data)
//        p.OutputDataReceived.Add(fun e -> printfn "Output %s" e.Data)
//        p.Start() |> ignore
//        p.WaitForExit()
//        p.ExitCode

// 2 minutes
//runProcess (Some(@"C:\EE\Git\TorchSharp2")) "dotnet" [||] "build"

// TODO Run Test Script


//#r @"C:\EE\Git\TorchSharp2\bin\AnyCPU.Debug\TorchSharp\netstandard2.0\TorchSharp.dll"
#r @"C:\EE\Git\TorchSharp2\bin\x64.Debug\Native\TorchSharp.dll"

TorchSharp.Tensor.TorchTensor.CodeGenTimestamp

(int) 637274133490606370L

//let [<Literal>] LibTorchSharp = @"C:\EE\Git\TorchSharp2\bin\x64.Debug\Native\LibTorchSharp.dll"
//
//[<DllImport(LibTorchSharp)>]
//extern int THSTensor_add_int(int left, int right);
//
//THSTensor_add_int(1,3)
