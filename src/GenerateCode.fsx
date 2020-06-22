#load "V2Parser.fsx"
#load "V2CodeGenLLTorchSharp.fsx"

// TODO - underscore for name before overload..
// TODO - figure out 64bit timestamp.... not important for now
// TODO Refactor first parameter
// TODO remove ? annotation from nullable reference types
// TODO use self assignment ??= to set empty arrays

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


open CodeGenCommon
open V2Parser

let noTensorOptions =
    set [ "zeros_like"; "empty_like"; "full_like"; "ones_like"; "rand_like"; "randint_like"; "randn_like"; ]

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

let methods() = 
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

let filterFunctions(fs : (string * ('a -> bool))[], key : 'a -> string) (xs: 'a[]) = 
    let filterSets = 
        fs 
        |> Array.map (fun (fName,f) -> (fName, xs |> Array.filter (f >> not) |> Array.map key |> Set)) 
        |> Array.sortByDescending (fun (_,xs) -> xs.Count)
    printfn "Total Unfiltered %i" xs.Length
    //let allSet = xs |> Array.map key |> Set
    let unfilteredSet = (xs,fs) ||> Array.fold (fun xs (_,f) -> xs |> Array.filter f) |> Array.map key |> Set
    //let filteredSet = Set.difference allSet unfilteredSet
    printfn "Total Filtered %i" unfilteredSet.Count
    printfn "Total filtered"
    for (name,set) in filterSets do
        printfn "%s %i" name set.Count
    printfn "Incremental filtered"
    filterSets |> Array.map (fun (name,_) -> 
       (xs,fs) 
       ||> Array.fold (fun xs (n,f) -> if n = name then xs else xs |> Array.filter f) 
       |> Array.map key |> Set 
       |> fun otherSet -> name, (otherSet.Count - unfilteredSet.Count))
    |> Array.sortByDescending snd
    |> Array.iter (fun (name,c) -> printfn "%s %i" name c)



let filterFuncs = 
    [|
        "Depricated",          (fun x -> not (x.overloadName = "deprecated" || x.depricated)) 
        "Exclude Prefixes",    (fun x -> excludedPrefixes |> Seq.exists (fun y -> x.name.StartsWith(y)) |> not)
        "Exclude Suffixes",    (fun x -> excludedSuffixes|> Seq.exists (fun y -> x.name.EndsWith(y)) |> not)
        "Generators",          (fun x -> not( x.overloadName.EndsWith("generator") || x.overloadName.EndsWith("generator_out")))
        "source",              (fun x -> x.overloadName.StartsWith("source_") |> not)
        "Excluded Functions",  (fun x -> excludedFunctions.Contains(x.name) |> not) // 20
        "Arg Types Dimname",   (fun x -> x.args |> Array.exists (fun x -> match x.type_ with | BT.Dimname | BT.DimnameList | BT.DimnameListOptional -> true | _ -> false) |> not)
        "Backward",            (fun x -> not(x.name.EndsWith("backward") || x.name.EndsWith("backward_out")))
        "Arg Types Generator", (fun x -> x.args |> Array.exists (fun x -> match x.type_ with | BT.Generator -> true | _ -> false) |> not)
        "Excluded Functions 2",(fun x -> aditionalFiltered.Contains(x.name,x.overloadName) |> not)
        "Arg Types Memory",    (fun x -> x.args |> Array.exists (fun x -> match x.type_ with | BT.MemoryFormat | BT.MemoryFormatOptional -> true | _ -> false) |> not)
//        "Code Gen Error",      (fun schema -> 
//                                    try
//                                        (schema |> V2CodeGenLLTorchSharp.genCpp, 
//                                         schema |> V2CodeGenLLTorchSharp.genHeader (true,true), 
//                                         schema |> V2CodeGenLLTorchSharp.genCSharp) |> ignore; 
//                                        true
//                                    with | _ -> false)
    |]


let schemas = 
    (V2Parser.schemas(),filterFuncs) 
    ||> Array.fold (fun xs (_,f) -> xs |> Array.filter f) 

// ScalarTypeOptional
//open V2CodeGenLLTorchSharp

//errorSchemas.[0].args.[5].Gated
//errorSchemas.[0].args.[5].defaultValue

//|> V2CodeGenLLTorchSharp.genCpp

//let schemas = V2Parser.schemas()
// TensorOptions

//schemas 
//|> Array.find (fun x -> x.name = "_sobol_engine_draw" && x.overloadName = "")



// Extra Extra filtered
//let schemas = extraFilteredSchemas() //Array.prepend (methods()) (extraFilteredSchemas())

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
                ""
            |]
        for schema in schemas do
            yield DLLImport
            yield schema |> V2CodeGenLLTorchSharp.genImport(true)
            yield ""
            yield! schema |> V2CodeGenLLTorchSharp.genCSharp
    |] 
    |> func ("public partial class TorchTensor : IDisposable")
    |> func ("namespace TorchSharp.Tensor")


File.WriteAllLines(CSharpTarget, [|yield! CSharpHeader; yield! CSharpBody|])
File.WriteAllLines(CPPTarget, [|yield! CPPHeader; yield! schemas |> Array.map V2CodeGenLLTorchSharp.genCpp |> concatWithNewLine|])
File.WriteAllLines(HTarget, [|yield! HeaderHeader; yield! schemas |> Array.map (V2CodeGenLLTorchSharp.genHeader (true,true) >> addFinalSemiColon) |> concatWithNewLine |])

