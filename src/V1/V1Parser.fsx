// NOTE This should probably be depricated...
#I @"..\bin\Debug\"
#r @"FParsecCS.dll"
#r @"FParsec.dll"

open System.IO
open FParsec

[<RequireQualifiedAccess>]
type Modifier = 
    /// int[] and int[2]
    | Array of int option 
    /// Tensor(a), Tensor(a!)
    | Alpha of char * bool
    /// Tensor?, 
    | Optional

[<RequireQualifiedAccess>]
type BaseType = 
    | Tensor
    | Bool
    | Int
    | Float
    | Scalar
    | ScalarType
    | Dimname
    | Astrix
    | Layout
    | Device
    | Generator
    | MemoryFormat
    | String
    | QScheme
    | Storage
    | ConstQuantizerPtr

[<RequireQualifiedAccess>]
type ParamType = 
    {
        baseType : BaseType
        name : string
        defaultValue : string option
        /// int[] and int[2]
        array : int option option
        /// Tensor(a), Tensor(a!)
        alpha : (char * bool) option
        /// Tensor?, 
        optional : bool
    }

let ws = spaces
let str s = pstring s
let numberInBrackets = between (str "[") (str "]") pint32

let identifier : Parser<string,unit> =
    let isIdentifierFirstChar c = isLetter c || c = '_'
    let isIdentifierChar c = isLetter c || isDigit c || c = '_'
    many1Satisfy2L isIdentifierFirstChar isIdentifierChar "identifier" .>> ws

let modifiers : Parser<Modifier list,unit> = 
    choice [
        (pchar '(' >>. asciiLetter .>>. (str "!)" <|> str ")") 
            |>> function 
                | c,")" -> Modifier.Alpha(c,false) 
                | c,"!)" -> Modifier.Alpha(c,true) 
                | _,_ -> failwith "err")
        str "?" |>> fun _ -> Modifier.Optional
        str "[]" |>> fun _ -> Modifier.Array(None)
        numberInBrackets |>> fun x -> Modifier.Array(Some(x))
    ] |> many

let tname : Parser<BaseType,unit> = 
    [
        "Tensor", BaseType.Tensor
        "bool", BaseType.Bool
        "int", BaseType.Int
        "float", BaseType.Float
        "ScalarType", BaseType.ScalarType
        "Scalar", BaseType.Scalar
        "Dimname", BaseType.Dimname
        "*", BaseType.Astrix
        "Generator", BaseType.Generator
        "Layout", BaseType.Layout
        "Device", BaseType.Device
        "MemoryFormat", BaseType.MemoryFormat
        "str", BaseType.String
        "QScheme", BaseType.QScheme
        "Storage", BaseType.Storage
        "ConstQuantizerPtr", BaseType.ConstQuantizerPtr
    ] 
    |> List.map (fun (x,y) -> str x |>> fun _ -> y)
    |> choice

let defaultValue : Parser<string,unit> = (many1SatisfyL (function | '_' | '-' | 'c' | 'e' | '.' -> true | c ->  isDigit c || isAsciiLetter c)  "defaultValue")
let typeSignature = ws >>. (tname .>>. modifiers .>> ws) .>>. opt identifier .>>. opt (pchar '=' >>. (defaultValue<|> (str "[]") <|> (str "[0,1]"))) .>> ws
let params_ = (sepBy typeSignature (pchar ','))

let funcParser = 
    ((sepBy1 identifier (pchar '.')) .>>. (pchar '(' >>. params_ .>> str ") ->") .>> ws) .>>. 
    ((between (pchar '(') (pchar ')') params_ ) 
        <|> (typeSignature |>> fun x -> [x]))

let parse (schema: string) = 
    match run funcParser schema with
    | ParserResult.Success(((names,inputs),outputs),_,_) -> 
        let f (((baseType: BaseType,modifiers: Modifier list),name: string option),default_: string option) :  ParamType =
            {
                baseType  = baseType
                array = modifiers |> List.choose (function | Modifier.Array(x) -> Some(x) | _ -> None) |> List.tryHead 
                alpha = modifiers |> List.choose (function | Modifier.Alpha(x,y) -> Some(x,y) | _ -> None) |> List.tryHead
                optional = modifiers |> List.exists ((=) Modifier.Optional)
                name = name |> Option.defaultValue ""
                defaultValue = default_
            } 
            
        ((match names with | [x] | [x;_] -> x | _ -> failwith "err"), 
         (match names with | [_] -> "" | [_;x] -> x | _ -> failwith "err"),
         inputs |> List.map f |> List.toArray,
         outputs |> List.map f |> List.toArray)
    | ParserResult.Failure(_) as x -> failwithf "%A" x
