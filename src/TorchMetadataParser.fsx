#I @".\bin\Debug\"
#r @"FParsecCS.dll"
#r @"YamlDotNet.dll"
#r @"FParsec.dll"

open YamlDotNet.RepresentationModel

open System.IO
open FParsec

type Dispatch = 
    | CPU
    | CUDA
    | MkldnnCPU
    | QuantizedCPU
    | QuantizedCUDA
    | SparseCPU
    | SparseCUDA
    static member TryParse(x: string) =
        match x with
        | "CPU" -> Some CPU 
        | "CUDA" -> Some CUDA 
        | "MkldnnCPU" -> Some MkldnnCPU
        | "QuantizedCPU" -> Some QuantizedCPU
        | "QuantizedCUDA" -> Some QuantizedCUDA
        | "SparseCPU" -> Some SparseCPU
        | "SparseCUDA" -> Some SparseCUDA
        | _ -> None



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
type Modifier = 
    /// int[] and int[2]
    | Array of int option 
    /// Tensor(a), Tensor(a!)
    | Alpha of char * bool
    /// Tensor?, 
    | Optional


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

[<RequireQualifiedAccess>]
type CombinedModifier = 
    {
        /// int[] and int[2]
        array : int option option
        /// Tensor(a), Tensor(a!)
        alpha : (char * bool) option
        /// Tensor?, 
        optional : bool
    }

type ParamType = 
    {
        baseType : BaseType
        modifiers : CombinedModifier
        name : string option
        default_ : string option
    }

type Attribute = 
    | Dispatch of (Dispatch*string)[]
    | CategoryOverrideFactory 
    | DeviceGuardFalse
    | ManualKernelRegistrationTrue
    | NamedGuardFalse
    | PythonModuleNN
    | RequiresTensorTrue
    | SupportsNamedTensorTrue
    | UseC10DispatcherFull
    | Variants of function_ : bool * method_ : bool
    //| Func of func : string * MetadataAST[]

type Func =  {
    firstName : string
    secondName : string
    inputs : ParamType[]
    outputs : ParamType[]
    attributes : Attribute[]
}

let loadSchemas(path:string) = 
    let ys = YamlStream()
    use sr = new StringReader(File.ReadAllText(path))
    ys.Load(sr)
    let doc = ys.Documents.[0]
    [|
        let rootNode = (doc.RootNode :?> YamlSequenceNode)
        let funcs = rootNode.Children
        for func in funcs do
            let func = (func :?> YamlMappingNode)
            let funcValue = ((func.Children |> Seq.find (fun x -> (x.Key :?> YamlScalarNode).ToString() = "func")).Value :?> YamlScalarNode).ToString()

            let attributes = 
                [|
                    for kv in func.Children do 
                        match (kv.Key :?> YamlScalarNode).ToString() with
                        | "dispatch" ->
                            yield Attribute.Dispatch([|for kv in (kv.Value :?> YamlMappingNode).Children -> 
                                                        Dispatch.TryParse(kv.Key.ToString()) |> Option.get, 
                                                        kv.Value.ToString()|])
                        | "func" -> ()
                        | x -> 
                            yield 
                                match x,(kv.Value :?> YamlScalarNode).ToString() with
                                | "use_c10_dispatcher", "full"-> UseC10DispatcherFull
                                | "variants", "function" -> Variants(true,false)
                                | "variants", "function, method" 
                                | "variants", "method, function" -> Variants(true,true)
                                | "variants", "method" -> Variants(false,true)
                                | "manual_kernel_registration", "True" -> ManualKernelRegistrationTrue
                                | "supports_named_tensor", "True" -> SupportsNamedTensorTrue
                                | "device_guard", "False" -> DeviceGuardFalse
                                | "requires_tensor", "True" -> RequiresTensorTrue
                                | "python_module", "nn" -> PythonModuleNN
                                | "category_override", "factory" -> CategoryOverrideFactory
                                | "named_guard", "False" -> NamedGuardFalse
                                | key,value -> 
                                    failwithf "Error, unexpected attribute %s or value %s" key value
                    |]
            match run funcParser funcValue with
            | ParserResult.Success(((names,inputs),outputs),_,_) -> 
                let f (((baseType: BaseType,modifiers: Modifier list),name: string option),default_: string option) :  ParamType =
                    {
                        baseType  = baseType
                        modifiers = 
                            { 
                                array = modifiers |> List.choose (function | Modifier.Array(x) -> Some(x) | _ -> None) |> List.tryHead 
                                alpha = modifiers |> List.choose (function | Modifier.Alpha(x,y) -> Some(x,y) | _ -> None) |> List.tryHead
                                optional = modifiers |> List.exists ((=) Modifier.Optional)
                            }
                        name = name
                        default_ = default_
                    } 
                    
                yield {
                    firstName = match names with | [x] | [x;_] -> x | _ -> failwith "err"
                    secondName = match names with | [_] -> "" | [_;x] -> x | _ -> failwith "err"
                    inputs = inputs |> List.map f |> List.toArray
                    outputs = outputs |> List.map f |> List.toArray
                    attributes = attributes
                }
            | ParserResult.Failure(_) as x -> failwithf "%A" x
            
    |] 

