#load "TargetModuleCPP.fsx"
#load "Clippy.fsx"
#load @"TorchMetadataParser.fsx"
//#load @"TorchMetadataExtensions.fsx"
#load @"DiffPlex.fsx"
#load @"CodeGenCommon.fsx"
open System
open Clippy
open DiffPlex
open TorchMetadataParser
open System.IO
open TargetModuleCpp
open CodeGenCommon
open CodeGenCommon.Cpp
open CodeGenCommon.CSharp

let [<Literal>] NNModulePrefix = "THSNN_"
let [<Literal>] EXPORT_API = "EXPORT_API"

// TODO fix non namespace function names to only be in the first name and not the second
// TODO ("","new_empty") figur out new Tensor, self?
// TODO newLong, also note at::ScalarType(at::kLong) instead of int64_t 
// TODO DiffPlex for https://github.com/mmanela/diffplex/
//
////let emptyModifiers : CombinedModifier = { array = None; alpha = None; optional = false }
////let alphaBangModifiers : CombinedModifier = { array = None; alpha = Some('a',true); optional = false }
//
//let simpleTensorOut : ParamType = { baseType = BaseType.Tensor;  name = ""; defaultValue = None; array = None; alpha = None; optional = false }
//
//let simpleTensorInput = {simpleTensorOut with  name = "input"}
//
////let schemas = loadSchemas(Path.Combine(__SOURCE_DIRECTORY__, "TorchMetadata.yaml"))
////
////let getSchmea(firstName,secondName) = 
////    schemas |> Array.find (fun x -> x.firstName = firstName && x.secondName = secondName)
////let searchSchemas(name: string) = 
////    schemas 
////    |> Array.filter (fun x -> x.firstName.Contains(name) || x.secondName.Contains(name)) 
////    |> Array.map (fun x -> x.firstName,x.secondName)
////
//


