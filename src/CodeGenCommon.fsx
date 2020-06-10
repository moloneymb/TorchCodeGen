open System

let [<Literal>] INDENT = "    "

module String = 
    let indent (n: int) (x: string) = x.Insert(0,String(Array.create n ' '))

let capitalizeFirst (x: string) = x |> String.mapi (function | 0 -> Char.ToUpperInvariant | _ -> id)

let asTuple(xs:string[]) = xs |> String.concat ", " |> sprintf "(%s)"

/// This appends a comma to a non-emepty string to for use in setting up parameters
let appendParam (x:string) = if String.IsNullOrWhiteSpace(x) then x else x + ", "
/// This appends a comma to a non-emepty string to for use in setting up parameters
let prependParam (x:string) = if String.IsNullOrWhiteSpace(x) then x else ", " + x

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

let multiLineParams(xs:string[]) = 
    xs |> Array.mapi (fun i x -> x + if xs.Length - 1 = i then ")" else ",")

let addFinalSemiColon(xs: string[]) = 
    xs |> Array.mapi (fun i x -> if xs.Length - 1 = i then x+";" else x)

let closeParen(xs: string[]) = 
    xs |> Array.mapi (fun i x -> if xs.Length - 1 = i then x+")" else x)

module Cpp = 
    // adds a semicolon to the last line
    let semicolon (xs:string[]) = match xs with | [||] -> [||] | _ -> [|yield! xs.[0..xs.Length - 2]; yield xs.[xs.Length-1] + ";"|]
    // Require new line
    let macro(name: string, break_ : bool) (lines: string[]) = 
        match lines, break_ with
        | [|x|], false -> [|sprintf "%s(%s)" name x|]
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

    let funcMany (functionLines: string[]) (body: string[]) = 
        match functionLines with
        | [||] -> failwith "err"
        | [|x|] -> func x body
        | xs -> [|yield xs.[0]; yield! xs.[1..] |> indent; yield "{"; yield! body |> indent; yield "}"|]
        
    let ternaryIfThenElse(conditional : string, then_: string, else_: string) = 
        sprintf "%s ? %s : %s" conditional then_ else_


module CSharp = 
    open Cpp
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

    let using(firstLine) (body:string[]) = func(sprintf "using (%s)" firstLine) body
    let unsafe (body: string[]) = func "unsafe" body 
    let fixed_(args: string) (body: string[]) = func (sprintf "fixed(%s)" args) body
    let nestedFixed (args: string[]) (body: string[]) =
        match args with
        | [||] -> body
        | [|x|] -> fixed_ x body
        | _ -> (args,body) ||> Array.foldBack (fun x body -> fixed_ x body) 

