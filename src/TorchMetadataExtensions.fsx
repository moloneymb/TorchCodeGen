#load "TorchMetadataParser.fsx"
open TorchMetadataParser

type BaseType with
    member this.BaseString = 
            match this with
            | BaseType.Tensor -> "Tensor"
            | BaseType.Bool -> "bool"
            | BaseType.Int -> "int"
            | BaseType.Float -> "float"
            | BaseType.Scalar -> "Scalar"
            | BaseType.ScalarType -> "ScalarType"
            | BaseType.Dimname -> "Dimname"
            | BaseType.Astrix -> "Astrix"
            | BaseType.Layout -> "Layout"
            | BaseType.Device -> "Device"
            | BaseType.Generator -> "Generator"
            | BaseType.MemoryFormat -> "MemoryFormat"
            | BaseType.String -> "String"
            | BaseType.QScheme -> "QScheme"
            | BaseType.Storage -> "Storage"
            | BaseType.ConstQuantizerPtr  -> "ConstQunatizerPtr"

    member this.BaseString2 = 
            match this with
            | BaseType.Tensor -> "Tensor"
            | BaseType.Bool -> "Bool"
            | BaseType.Int -> "Int"
            | BaseType.Float -> "Float"
            | BaseType.Scalar -> "Scalar"
            | BaseType.ScalarType -> "ScalarType"
            | BaseType.Dimname -> "Dimname"
            | BaseType.Astrix -> "Astrix"
            | BaseType.Layout -> "Layout"
            | BaseType.Device -> "Device"
            | BaseType.Generator -> "Generator"
            | BaseType.MemoryFormat -> "MemoryFormat"
            | BaseType.String -> "String"
            | BaseType.QScheme -> "QScheme"
            | BaseType.Storage -> "Storage"
            | BaseType.ConstQuantizerPtr  -> "ConstQunatizerPtr"

[<RequireQualifiedAccess>]
type ScalarType = 
    | Byte = 0
    | Char = 1
    | Short = 2
    | Int = 3
    | Long = 4
    | Half = 5
    | Float = 6
    | Double = 7
    | ComplexHalf = 8
    | ComplexFloat = 9
    | ComplexDouble = 10

let scalarTypeToCppType(x: ScalarType) = 
    match x with
    | ScalarType.Byte -> "uint8_t"
    | ScalarType.Char -> "int8_t"
    | ScalarType.Short -> "int16_t"
    | ScalarType.Int -> "int"
    | ScalarType.Long -> "int64_t"
    | ScalarType.Half -> "at::Half"
    | ScalarType.Float -> "float"
    | ScalarType.Double -> "double"
    | ScalarType.ComplexHalf -> "at::ComplexHalf"
    | ScalarType.ComplexFloat -> "std::complex<float>"
    | ScalarType.ComplexDouble -> "std::complex<double>"
    | _ -> failwith "err"

let scalarTypeToString(x: ScalarType) = 
    match x with
    | ScalarType.Byte -> "Byte"
    | ScalarType.Char -> "Char"
    | ScalarType.Short -> "Short"
    | ScalarType.Int -> "Int"
    | ScalarType.Long -> "Long"
    | ScalarType.Half -> "Half"
    | ScalarType.Float -> "Float"
    | ScalarType.Double -> "Double"
    | ScalarType.ComplexHalf -> "ComplexHalf"
    | ScalarType.ComplexFloat -> "ComplexFloat"
    | ScalarType.ComplexDouble -> "ComplexDouble"
    | _ -> failwith "err"
    
let scalars = 
    [|
        "SByte", ScalarType.Byte
        "Byte", ScalarType.Byte
        "Short", ScalarType.Short
        "Int", ScalarType.Int
        "Long", ScalarType.Long
        "Float", ScalarType.Float
        "Double", ScalarType.Double
    |]

type ParamType with
    member this.BaseString =
        let array = 
            match this.array with
            | None -> ""
            | Some(None) -> "[]"
            | Some(Some(x)) -> sprintf "[%i]" x
        let alpha = 
            match this.alpha with
            | None -> ""
            | Some(c,bang) -> sprintf "(%c%s)" c (if bang then "!" else "")
        let optional = if this.optional then "?" else ""
        alpha + array + optional

    /// This is a simple parameter
    member this.IsBase = 
        this.array = None && this.optional = false && this.defaultValue = None && this.alpha = None
    member this.IsSimpleAlpha = this.IsBase && this.alpha = Some('a',true)
    member this.IsArray = this.array.IsSome

let (|Function|_|) (x: Func) = 
    x.attributes 
    |> Array.exists (function | Variants(true,_) -> true | _ -> false)
    |> function | true -> Some(x) | false -> None

let (|Method|_|) (x: Func) = 
    x.attributes 
    |> Array.exists (function | Variants(_,true) -> true | _ -> false)
    |> function | true -> Some(x) | false -> None

let (|FunctionAndMethod|_|) (x: Func) = 
    match x with
    | Method(_) | Function(_) -> Some(x)
    | _ -> None

type Func with
    member this.IsFixedOutput = this.outputs |> Array.exists (fun x -> x.IsArray) |> not
