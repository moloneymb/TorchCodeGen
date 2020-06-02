open System
open System.IO

let camelToUnderscore = 
    let r = System.Text.RegularExpressions.Regex("(?!^)([A-Z])([a-z])")
    fun (name:string) -> 
        r.Replace(name,"_${1}${2}").ToLowerInvariant()

let underscoreToCamel(name:string) = 
    let c1 = 
        name.ToCharArray() 
        |> Array.fold (fun (lastUnderscore: bool ,xs:char list) (c:char) -> 
            match c with
            | '_' -> (true,xs)
            | _ -> (false, (if lastUnderscore then Char.ToUpperInvariant(c) else c) :: xs)
            ) (false,[])
            |> snd |> List.toArray |> Array.rev |> String
    if name.EndsWith("_") then c1 + "_" else c1
    
let baseDir = @"C:\EE\Git\TorchSharp\src\" 
let TorchSharpDir = Path.Combine(baseDir,"TorchSharp")
let NativeDir = Path.Combine(baseDir,"Native", "LibTorchSharp")
let THSNNh = Path.Combine(NativeDir, "THSNN.h")
let THSTensorh = Path.Combine(NativeDir, "THSTensor.h")
let THSNNcpp = Path.Combine(NativeDir, "THSNN.cpp")

let [<Literal>] NNModulePrefix = "THSNN_"
let [<Literal>] EXPORT_API = "EXPORT_API"
