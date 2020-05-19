#load "TargetModuleCPP.fsx"
#load "Clippy.fsx"
#load @"TorchMetadataParser.fsx"
#load @"TorchMetadataExtensions.fsx"
#load @"DiffPlex.fsx"
open System
open Clippy
open TorchMetadataParser
open TorchMetadataExtensions
open DiffPlex
open System.IO
open TargetModuleCpp

// TODO fix non namespace function names to only be in the first name and not the second
// TODO ("","new_empty") figur out new Tensor, self?
// TODO newLong, also note at::ScalarType(at::kLong) instead of int64_t 
// TODO DiffPlex for https://github.com/mmanela/diffplex/

let schemas = loadSchemas(Path.Combine(__SOURCE_DIRECTORY__, "TorchMetadata.yaml"))

let getSchmea(firstName,secondName) = 
    schemas |> Array.find (fun x -> x.firstName = firstName && x.secondName = secondName)


//let emptyModifiers : CombinedModifier = { array = None; alpha = None; optional = false }
//let alphaBangModifiers : CombinedModifier = { array = None; alpha = Some('a',true); optional = false }

let simpleTensorOut : ParamType = { baseType = BaseType.Tensor;  name = ""; defaultValue = None; array = None; alpha = None; optional = false }
let simpleTensorInput = {simpleTensorOut with  name = "input"}

let searchSchemas(name: string) = 
    schemas 
    |> Array.filter (fun x -> x.firstName.Contains(name) || x.secondName.Contains(name)) 
    |> Array.map (fun x -> x.firstName,x.secondName)

let printAsTuple(xs:string[]) = xs |> String.concat ", " |> sprintf "(%s)"

let printParamTypes(xs:ParamType[]) = 
    [|for x in xs -> 
       sprintf  "%s%s%s%s" 
            x.baseType.BaseString 
            x.BaseString 
            (match x.name with | "" -> "" | x ->" " + x)
            (match x.defaultValue with | None -> "" | Some(x) -> "=" + x)
    |]
    |> printAsTuple

let printSchemaSimple(schema: Func) = 
    // NOTE: This ignores attributes for now
    printfn "%s.%s%s -> %s" 
        schema.firstName 
        schema.secondName 
        (printParamTypes(schema.inputs))
        (printParamTypes(schema.outputs))

let [<Literal>] NNModulePrefix = "THSNN_"
let [<Literal>] EXPORT_API = "EXPORT_API"
let [<Literal>] INDENT = "    "


module String = 
    let indent (n: int) (x: string) = x.Insert(0,String(Array.create n ' '))

let capitalizeFirst (x: string) = x |> String.mapi (function | 0 -> Char.ToUpperInvariant | _ -> id)

module Array = 
    let concatWith (xs:seq<'a[]>) (ys:'a[]) = 
        let rec f (xs:List<'a[]>) : 'a[] = 
            [|
                match xs with
                | [] -> ()
                | [xs] -> yield! xs
                | (x::xs) -> yield! x; yield! ys; yield! f xs
            |]
        f (xs |> Seq.toList)

let concatWithNewLine (xs:string[][]) : string[] = Array.concatWith xs [|""|]

let indent (xs: string[]) = xs |> Array.map (fun x -> INDENT + x)

// Require new line
let macro(name: string, break_ : bool) (lines: string[]) = 
    match lines, break_ with
    | [|x|], false -> [|sprintf "%s(%s);" name x|]
    | _,true -> [|yield sprintf "%s(" name; yield! lines |> indent; yield ")"|]
    | _, false -> failwith "err" // multi-line must break

let ifThenElse(conditional : string, then_: string[], else_: string[]) = 
    [|
        yield sprintf "if (%s)" conditional
        yield "{"
        yield! then_ |> indent
        match else_ with
        | [||] -> () 
        | _ -> 
            yield "} else {"
            yield! else_ |> indent
        yield "}"
    |]

let ifThen(conditional : string) (then_: string[]) = ifThenElse(conditional, then_, [||])

let func(firstLine) (body: string[]) = 
    [| yield firstLine; yield "{"; yield! body |> indent; yield "}" |]

let using(firstLine) (body:string[]) = func(sprintf "using (%s)" firstLine) body

module CSharp = 
    let extern_(body: string) = [| "[DllImport (\"LibTorchSharp\")]"; sprintf "extern static %s" body|]
    let namespace_(namespace_: string) (body: string[]) =
        [| 
            yield sprintf "namespace %s" namespace_
            yield "{"
            yield! body |> indent
            yield "}"
        |]

    let getSetMember(fistLine: string,get: string[],set: string[]) =
        [| 
            yield fistLine + " {"
            yield! [| yield "get {"; yield! get |> indent; yield "}" |] |> indent
            yield! [| yield "set {"; yield! set |> indent; yield "}" |] |> indent
            yield "}"
        |]

/// This appends a comma to a non-emepty string to for use in setting up parameters
let appendParam (x:string) = if String.IsNullOrWhiteSpace(x) then x else x + ", "
/// This appends a comma to a non-emepty string to for use in setting up parameters
let prependParam (x:string) = if String.IsNullOrWhiteSpace(x) then x else ", " + x
