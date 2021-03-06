﻿// NOTE: We can use Null pointer parameters :)

// TODO ScalarType? type = null, 
//    type.HasValue, (sbyte)type.GetValueOrDefault())

// Learings
// Tensor and Scalar , e.g. bias ? *bias : at::Tensor() and for at::Scalar() etc.
// If it's nullable we could just pass in a null pointer instead of guarding it with a with statement
// Method of namespace torch::conv_transpose2d


// Some missing methods with no obvious location for name
//  [|(DimnameListOptional, 10); (ScalarTypeOptional, 1); (TensorList, 4); (IntOptional, 6); (ScalarOptional, 9)|]
//   schemas |> Array.collect (fun x -> (x.args |> Array.filter (fun x -> x.dynamicType <> BT.Tensor && x.isNullable && x.defaultValue.IsNone))) |> Array.map (fun x -> x.type_) |> Array.countBy id

#load "V2Parser.fsx"
//#load "V2Filtered.fsx"
//#load "DiffPlex.fsx"
//#load "Clippy.fsx"
//#load "TargetModuleCPP.fsx"
#load "Common.fsx"
open Common
open System
open CodeGenCommon
open V2Parser
//open TargetModuleCpp
//open Clippy
//open DiffPlex
open CSharp
open Cpp
open System
open System.IO

let THSTensor = "THSTensor__"

let MaxLineWidth = 120
let compressLines = compressLines MaxLineWidth

let TensorAllocator = "Tensor* (*allocator)(size_t length)"
let TensorAllocatorCSharp = "AllocatePinnedArray allocator"

let toCSharpName(name: string) = 
    name |> underscoreToCamel |> CSharp.sainitize

//"""
//var res = THSTensor___cast_Byte(Handle,
//        nonBlocking.HasValue,
//        nonBlocking.HasValue ? nonBlocking.Value : false);
//"""
//

/// A rule of thumb to determine if an argument should be gated
/// or Null should be passed through
let canBeNull(t: BT) = 
    if t.IsArray then true
    else 
        match t with
        | TensorOrScalar _ -> true
        | _ -> false

type Arg with

    member x.Gated = 
        if canBeNull x.type_ then false
        else
        //not(canBeNull x.type_) && 
            (x.isNullable ||  x.defaultValue.IsSome) &&
            not(x.name = "options" && (match x.type_ with | BT.TensorOptions | BT.TensorOptionsAnd -> true | _ -> false))

    member x.IsOptional = x.isNullable || x.defaultValue.IsSome

    member arg.CSharpType = 
        match arg.type_ with
        | BT.TensorOptions
        | BT.TensorOptionsAnd when arg.name = "options" -> 
            failwith "this should be filtered out and handled elsewhere"
        | BT.Bool2 | BT.Bool3 | BT.Bool4 -> failwithf "todo %A" arg
        | BT.IntList -> "long[]"
        | BT.TensorList (* when not x.isNullable *) -> "TorchTensor[]"
        | BT.DimnameList -> failwithf "todo %A" arg 
        | BT.String -> "string"
        | _ when arg.type_.IsArray -> failwithf "err %A" arg
        | _ ->
            match arg.type_ with
            | BT.Bool -> "bool" 
            | BT.Int -> "long"
            | BT.IntOptional-> "long"
            | BT.Double -> "double"
            | BT.DoubleOptional -> "double"
            | BT.Tensor -> "TorchTensor"
            | BT.TensorAnd -> "TorchTensor"
            | BT.ConstTensor -> "TorchTensor"
            | BT.ScalarType 
            | BT.ScalarTypeOptional -> "ScalarType"
            | BT.Device -> failwith "todo"
            | BT.Scalar -> "Scalar"
            | BT.ScalarOptional -> "Scalar" 
            | BT.MemoryFormat 
            | BT.MemoryFormatOptional -> "MemoryFormat" // unsure about this
            | _ -> failwithf "todo %A" arg

    member arg.CSignature = 
        //let argName = arg.name |> toCSharpName
        match arg.type_ with
        | BT.TensorOptions
        | BT.TensorOptionsAnd when arg.name = "options" -> "const int8_t scalar_type, const char* device, const bool requires_grad"
        | BT.TensorOptions
        | BT.TensorOptionsAnd -> sprintf "int %s_kind, int %s_device" arg.name arg.name

        | BT.Bool2 | BT.Bool3 | BT.Bool4 -> failwithf "todo %A" arg
        | BT.IntList -> sprintf "const int64_t* %s, const int %s_length" arg.name arg.name
        //| BT.TensorList when x.isNullable -> sprintf "int %s_kind, int %s_device" name name
        | BT.TensorList (* when not x.isNullable *) -> sprintf "const Tensor* %s, const int %s_length" arg.name arg.name
        | BT.DimnameList -> failwithf "todo %A" arg
        | BT.String -> Printf.sprintf "const char* %s" arg.name

            //| BaseType.Strig -> 
            //    sprintf "char* %s_ptr, int %s_len" name name
        | _ when arg.type_.IsArray -> failwithf "err %A" arg
        | _ ->
            match arg.type_ with
            | BT.Bool -> "const bool" // when is this bool or int??
            | BT.Int
            | BT.IntOptional-> "const int64_t"
            | BT.Double -> "double"
            | BT.DoubleOptional -> "double"
            | BT.Tensor -> "Tensor"
            | BT.TensorAnd -> "const Tensor"
            | BT.ConstTensor -> "const Tensor"
            | BT.ScalarType 
            | BT.ScalarTypeOptional -> "const int8_t"
            //| BT.Device -> "int"
            | BT.Device -> "const int8_t" // unsure about this...
            | BT.Scalar -> "const Scalar"
            | BT.ScalarOptional -> "const Scalar" 
            | BT.MemoryFormat 
            | BT.MemoryFormatOptional -> "const MemoryFormat"
            | _ -> (string arg.type_) //failwithf "err %A" x
            |> fun x -> sprintf "%s %s" x arg.name 
        |> fun x -> 
            if arg.Gated 
            then sprintf "const bool with_%s, %s" arg.name x
            else x

    member arg.CSharpInteropSignature = 
        let argName = arg.name |> toCSharpName
        match arg.type_ with
        | BT.TensorOptions
        | BT.TensorOptionsAnd when arg.name = "options" -> 
            "sbyte scalarType, [MarshalAs(UnmanagedType.LPStr)] string device, bool requiresGrad"
//        | BT.TensorOptions
//        | BT.TensorOptionsAnd -> sprintf "int %s_kind, int %s_device" argName argName
        | BT.Bool2 | BT.Bool3 | BT.Bool4 -> failwithf "todo %A" arg
        | BT.IntList -> sprintf "IntPtr %s, int %s_length" argName argName
        //| BT.TensorList when x.isNullable -> sprintf "int %s_kind, int %s_device" name name
        | BT.TensorList (* when not x.isNullable *) -> sprintf "IntPtr %s, int %s_length" argName argName
        | BT.DimnameList -> failwithf "todo %A" arg
        | _ when arg.type_.IsArray -> failwithf "err %A" arg
        | _ ->
            match arg.type_ with
            | BT.String -> "[MarshalAs(UnmanagedType.LPStr)] string"
            | BT.Bool -> "bool" // when is this bool or int??
            | BT.Int
            | BT.IntOptional-> "long"
            | BT.Double -> "double"
            | BT.DoubleOptional -> "double"
            | BT.Tensor -> "IntPtr"
            | BT.TensorAnd -> "IntPtr"
            | BT.ConstTensor -> "IntPtr"
            | BT.ScalarType 
            | BT.ScalarTypeOptional -> "sbyte"
            //| BT.Device -> "int"
            | BT.Device -> "sbyte" // unsure about this...
            | BT.Scalar -> "IntPtr"
            | BT.ScalarOptional -> "IntPtr" 
            | BT.MemoryFormat 
            | BT.MemoryFormatOptional -> "MemoryFormat" // unsure about this
            | _ -> (string arg.type_) //failwithf "err %A" x
            |> fun x -> sprintf "%s %s" x argName 
        |> fun x -> 
            if arg.Gated 
            then sprintf "bool with_%s, %s" argName x
            else x

    member arg.CppToC =
        if arg.type_.IsArray then
            match arg.type_ with
            | BT.IntList -> "at::ArrayRef<int64_t>" + (sprintf "(%s, %s_length)" arg.name arg.name)
            | BT.TensorList -> "toTensors<at::Tensor>((torch::Tensor**)"  + (sprintf "%s, %s_length)" arg.name arg.name)
            //| BT.DimnameListOptional -> // NOTE: need to pull in additional headers to use Dimname
            | _ -> failwithf "todo cppToC %A " arg
            
        else
            match arg.type_ with
            | TensorOrScalar _ -> "*" + arg.name
            | _ -> arg.name
        |> fun y -> 
            let defaultValue = 
                arg.defaultValue 
                |> Option.map (fun x ->
                    match arg.type_,x with
                    | Tensor _, "{}" -> "at::Tensor()"
                    | BT.Scalar, "{}"
                    | BT.ScalarOptional,"{}" -> "at::Scalar()"
                    | _ ,_ -> x)
            let y = 
                match arg.type_ with
                | BT.ScalarOptional -> 
                     sprintf "c10::optional<c10::Scalar>(%s)" y // Note: it's converted from at::Scalar to c10::Scalar
                | BT.IntOptional -> 
                     sprintf "c10::optional<int64_t>(%s)" y
                | BT.DoubleOptional -> 
                     sprintf "c10::optional<double>(%s)" y
                | BT.ScalarType -> sprintf "c10::ScalarType(%s)" y
                | BT.ScalarTypeOptional -> sprintf "c10::optional<c10::ScalarType>(c10::ScalarType(%s))" y
                | BT.String -> sprintf "std::string(%s)" y
                | _ -> y
            if arg.Gated then
                match arg.type_ with
                | BT.ScalarTypeOptional -> 
                    ternaryIfThenElse(sprintf "with_%s" arg.name, y, "c10::nullopt")
                | _ -> 
                    ternaryIfThenElse(sprintf "with_%s" arg.name, y, defaultValue.Value)
            else
                if (match arg.type_ with | BT.TensorOptions | BT.TensorOptionsAnd -> true | _ -> false) && arg.name = "options" then
                    y
                else
                    if arg.isNullable && arg.defaultValue.IsSome then
                        ternaryIfThenElse(arg.name, y, defaultValue.Value)
                    else y
    member this.IsTensor = match this.type_ with | Tensor _ -> true | _ -> false
    member this.IsTensorOrScalar = match this.type_ with | TensorOrScalar _ -> true | _ -> false
    member this.ParameterNullable = this.isNullable || this.defaultValue.IsSome 
    /// Not wrappable in Nullable<T>
    member this.Nullableable = 
        match this.type_ with
        | BT.IntList 
        | TensorOrScalar _ -> false
        | _ -> true


    /// Wrappable in a Nullable<T> object in a parameter
    /// This is expected to be expanded
//    let isNullableable(arg: Arg) = 
type BT with
    member x.ReqPinning = 
        x.IsArray || 
            match x with
            | _ -> false // TODO... 

[<RequireQualifiedAccess>]
type RT = 
 | Empty
 | Single of BT
 | SingleTensor
 | SingleScalar
 | TensorTuple of int
 | ManyTensor
 with 
    member this.HasAllocator = 
        match this with
        | RT.TensorTuple _ 
        | RT.ManyTensor -> true
        | _ -> false

    member this.Return = 
        match this with
        | RT.SingleTensor
        | RT.Single BT.Tensor -> "Tensor"
        | RT.SingleScalar
        | RT.Single BT.Scalar -> "Scalar"
        | RT.Single BT.Bool -> "bool" // TODO revist this, should it be int
        | RT.Single BT.Int -> "int" 
        | RT.Single BT.Double -> "double" 
        | RT.Empty
        | RT.ManyTensor
        | RT.TensorTuple _ -> "void"
        | RT.Single BT.ScalarType -> "int8_t"
        | _ -> failwithf "TODO %A" this

type Schema with
    member this.Return =
        match this.returns with
        | [||] -> RT.Empty
        | [|x|] -> 
            match x.dynamicType with
            | BT.Tensor -> RT.SingleTensor
            | BT.Scalar -> RT.SingleScalar
            | BT.Bool -> RT.Single(BT.Bool)
            | BT.Int -> RT.Single(BT.Int)
            | BT.Double -> RT.Single(BT.Double)
            | BT.ScalarType -> RT.Single(BT.ScalarType)
            | BT.TensorVector
            | BT.TensorList -> RT.ManyTensor
            | _ -> failwith "todo"
        | xs when (xs |> Array.exists (fun x -> x.dynamicType <> BT.Tensor) |> not) -> //(x.dynamicType = BT.Tensor && y.dynamicType = BT.Tensor -> 
             RT.TensorTuple xs.Length
        | _ -> failwith "todo"

    member this.FunctionName = 
        this.name + this.overloadName + (if this.name.EndsWith("_") then "_" else "")

    member this.HasOptions = 
        this.args |> Array.exists (fun x -> 
            match x.type_ with 
            | BT.TensorOptions
            | BT.TensorOptionsAnd when x.name = "options" -> true
            | _ -> false)

    member schema.IsInstanceAndSelfName = 
        let isInstanceMember = schema.methodOfTensor = Some(true)
        let name = 
            if not isInstanceMember then None
            else 
                // it appears that instance memebers may not be the first argument (e.g. polygamma)
                // special casing for the second argument
                // TODO maybe generalize this
                match schema.args |> Array.toList with
                | [] -> None
                | (x::xs) when (match x.type_ with | Tensor _ -> true | _ -> false) -> Some(schema.args.[0].name)
                | (_::x::xs) when (match x.type_ with | Tensor _ -> true | _ -> false) ->  Some(schema.args.[1].name)
                | _ ->  None
        (isInstanceMember,name)

let genImport(singleLine: bool) (schema: Schema): string  = 
    let ret = 
        match schema.Return with
        | RT.Empty -> "void"
        | RT.Single BT.Bool -> "bool"
        | RT.Single BT.Int -> "int"
        | RT.Single BT.Double -> "double"
        | RT.Single BT.ScalarType -> "sbyte"
        | RT.SingleScalar
        | RT.SingleTensor -> "IntPtr"
        | RT.ManyTensor -> "void"
        | RT.TensorTuple _ -> "void"
        | _ -> failwithf "todo return name %A" schema.Return

    let args = 
        schema.args |> Array.map (fun x -> x.CSharpInteropSignature)
        |> fun xs -> if schema.Return.HasAllocator then Array.append [|TensorAllocatorCSharp;|] xs else xs //[|yield xs.[0]; yield  yield! xs.[1..]|] else xs
    let firstLine = 
        sprintf "private static extern %s %s%s(" ret THSTensor schema.FunctionName 
    if singleLine 
    then sprintf "%s%s);" firstLine (args |> String.concat ", ")
    else sprintf "%s%s    %s);" firstLine Environment.NewLine (args |> String.concat (sprintf ",%s    " Environment.NewLine))


//|> fun x -> sprintf "%s %s" x name 
//    |> fun y -> 
//        if arg.Gated 
//        then y + "?"
//        else y
//topk.args |> Array.map getCSharpArg

let genHeader (singleLine: bool, forExport: bool) (schema: Schema) : string[] = 
    let r = schema.Return
    let overloadedName = schema.FunctionName
    let args = 
        schema.args |> Array.map (fun x -> x.CSignature)
        |> fun xs -> if r.HasAllocator then Array.append [|TensorAllocator|] xs (*[|yield xs.[0]; yield TensorAllocator; yield! xs.[1..]|]*) else xs
    let returnWithExport = 
        if forExport then sprintf "EXPORT_API(%s)" r.Return
        else r.Return
    if singleLine then
        [|sprintf "%s %s%s(%s)" returnWithExport THSTensor overloadedName (args |> String.concat ", ")|]
    else 
        [|
            yield sprintf "%s %s%s(" returnWithExport THSTensor  overloadedName 
            yield! 
                match args |> multiLineParams with
                | [||] -> [|")"|]
                | xs -> xs
        |] 

let genCpp(schema: Schema) : string[] = 
    let r = schema.Return
    let first,tailArgs = 
        match schema.methodOfNamespace, schema.methodOfTensor with
        | Some(true),_ -> (sprintf "torch::%s(" schema.name, schema.args)
        | _, Some(true) -> ((sprintf "%s->%s(") schema.args.[0].name schema.name, schema.args.[1..])
        | _, _ -> failwith "err - come back to this if needed"

    let options = [|
        "auto options = at::TensorOptions()"
        "    .dtype(at::ScalarType(scalar_type))"
        "    .device(device)"
        "    .requires_grad(requires_grad);" |]
   
    [|
        let xs = tailArgs |> Array.map (fun x -> x.CppToC) 
        let singleLine = xs |> String.concat ", "
        let manyLines = 
            schema.args |> Array.map (fun x -> x.CppToC) |> multiLineParams |> indent 
        if schema.HasOptions 
        then 
            yield! options
        match r with
        | RT.SingleScalar -> 
            if singleLine.Length < MaxLineWidth then
                yield sprintf "CATCH_SCALAR(%s%s))" first singleLine
            else
                yield!
                    [|
                        yield sprintf "    %s" first
                        yield! manyLines |> addFinalSemiColon
                    |]
                    |> macro("CATCH_SCALAR",true)
        | RT.SingleTensor -> 
            if singleLine.Length < MaxLineWidth then
                yield sprintf "CATCH_TENSOR(%s%s))" first singleLine
            else
                yield!
                    [|
                        yield sprintf "    %s" first
                        yield! manyLines |> addFinalSemiColon
                    |]
                    |> macro("CATCH_TENSOR",true)
        | RT.TensorTuple c-> 
            yield!
                [|
                    yield sprintf "auto res = %s" first
                    yield! manyLines |> addFinalSemiColon
                    yield sprintf "Tensor * result = allocator(%i);" c
                    for i in 0..c-1 do
                        yield sprintf "result[%i] = new torch::Tensor(std::get<%i>(res));" i i
                |] //|> macro("CATCH",true)
        | RT.ManyTensor ->
            yield!
                [|
                    yield sprintf "auto res = %s" first
                    yield! manyLines |> addFinalSemiColon
                    yield "const size_t sz = res.size();"
                    yield "Tensor * result = allocator(sz);"
                    yield "for (size_t i = 0; i < sz; i++)"
                    yield "    result[i] = new torch::Tensor(res[i]);"
                |] //|> macro("CATCH",true)
        | RT.Empty -> 
            if singleLine.Length < MaxLineWidth then
                yield sprintf "CATCH(%s%s))" first singleLine
            else
                yield!
                    [| yield sprintf "    %s" first; yield! manyLines |> closeParen |> addFinalSemiColon |]
                    |> macro("CATCH",true)
        | RT.Single BT.Int
        | RT.Single BT.Double
        | RT.Single BT.Bool -> 
            if singleLine.Length < MaxLineWidth then
                yield sprintf "return %s%s);" first singleLine
            else
                yield sprintf "    return %s" first; 
                yield! manyLines |> addFinalSemiColon
        | RT.Single BT.ScalarType -> 
            if singleLine.Length < MaxLineWidth then
                yield sprintf "return (int8_t) %s%s);" first singleLine
            else
                yield sprintf "    return (int8_t) %s" first; 
                yield! manyLines |> addFinalSemiColon
        | _ -> failwithf "todo %A" r
    |]
    |> fun body -> 
        let singleLine = (genHeader (true,false) schema).[0] 
        if singleLine.Length < MaxLineWidth then func singleLine body
        else funcMany (genHeader (false,false) schema) body


let genCSharp(schema: Schema) : string[] = 
    let rt = 
        match schema.Return with
        | RT.Empty -> "void"
        | RT.SingleTensor -> "TorchTensor"
        | RT.SingleScalar -> "Scalar"
        | RT.Single BT.Bool -> "bool"
        | RT.Single BT.Int -> "int"
        | RT.Single BT.Double -> "double"
        | RT.Single BT.ScalarType -> "ScalarType"
        | RT.Single x -> failwithf "return type not yet supported %A" x
        | RT.TensorTuple n -> 
            [|for x in schema.returns -> sprintf "TorchTensor %s" (x.name |> toCSharpName)|] 
            |> String.concat ", " |> sprintf "(%s)"
        | RT.ManyTensor -> "TorchTensor[]"
    
    let isInstanceMember,selfName = schema.IsInstanceAndSelfName
    let args = schema.args

//    let isInstanceMember = schema.methodOfTensor = Some(true)
//    let args = 
//        if not isInstanceMember then schema.args
//        else 
//            // it appears that instance memebers may not be the first argument (e.g. polygamma)
//            match schema.args |> Array.toList with
//            | [] -> [||]
//            | (x::xs) when (match x.type_ with | Tensor _ -> true | _ -> false) -> schema.args.[1..]
//            | (_::x::xs) when (match x.type_ with | Tensor _ -> true | _ -> false) -> [|yield schema.args.[0]; yield! schema.args.[2..]|]
//            | _ -> schema.args

//        schema.args 
//        |> Array.tryHead 
//        |> Option.map (fun x -> match x.type_ with | Tensor _ -> true | _ -> false) 
//        |> Option.defaultValue false
//        |> function | true -> true,schema.args.[1..] | _ -> false,schema.args
    // Checking to see if we have optional after non-optional and we do... so this is commented out now
//    (false,schema.args) 
//    ||> Array.fold (fun isOptional (x:Arg) -> 
//        if (x.isNullable || x.defaultValue.IsSome) then true 
//        else (if isOptional then failwith "Optional arguments should appear after all required arguments" else false)) 
//    |> ignore<bool>
        
    let getCSharpArg(arg: Arg) : string = 
        let argType = arg.CSharpType
        let name = (arg.name |> toCSharpName)
        match arg.ParameterNullable, arg.Nullableable with 
        | true,true-> sprintf "%s? %s = null" argType  name 
        | true,false -> sprintf "%s %s = null" argType name 
        | false,_ -> sprintf "%s %s" argType name

    let parameters = 
        args 
        |> Array.partition (fun x -> x.IsOptional) 
        |> fun (xs,ys) -> [| yield! ys; yield! xs |]
        |> Array.filter (fun x -> 
            match x.type_ with 
            | BT.TensorOptions
            | BT.TensorOptionsAnd when x.name = "options" -> false
            | _ -> true)
        |> Array.map getCSharpArg 
        |> fun xs -> 
                if not schema.HasOptions then xs
                else 
                    xs |> Array.prepend [|
                        "ScalarType dtype = ScalarType.Float"
                        "string device = \"cpu\""
                        "bool requiresGrad = true"
                    |]
        |> String.concat ", "
    let name = (schema.name|> toCSharpName) // NOTE: Not using overload
    let firstLine = 
        sprintf "public %s%s %s(%s)" 
            (if not isInstanceMember then "static " else "") rt name parameters
    let anyPinning = args.Length > 0 && args |> Array.exists (fun x -> x.type_.ReqPinning)
    let isUnsafe = anyPinning // || ... todo other reasons
    let hasAllocator = match schema.Return with | RT.TensorTuple _ | RT.ManyTensor -> true | _ -> false
    
    let cArgs = 
        [| 
            // This complication is arond the allocator function, instead we will add it to the first parameter
            if hasAllocator then yield "pa.CreateArray"
            //elif args.Length > 0 then yield sprintf "%s.Handle" args.[0].name |> toCSharpName 
            for arg in args do
                match arg.type_ with
                | BT.TensorOptions
                | BT.TensorOptionsAnd when arg.name = "options" -> ()
                | _ -> 
                    if isInstanceMember && Some(arg.name) = selfName then 
                        yield "Handle" 
                    else
                        let argName = arg.name |> toCSharpName 
                        let conversion = 
                            match arg.type_ with
                            | BT.ScalarType
                            | BT.ScalarTypeOptional-> "(sbyte) "
                            //| BT.TensorList 
                            //| BT.TensorVector -> "(IntPtr) "
                            | _ -> ""
                        if arg.IsTensorOrScalar then 
                            match arg.ParameterNullable, arg.Nullableable, arg.type_.IsArray with
                            | _,_,true-> yield sprintf "(IntPtr) p%s" argName
                            | true,false,_ -> yield sprintf "%sPtr" argName
                            | _,_,false -> yield argName + ".Handle"
                        elif arg.Gated then 
                            // This is based on type and is completely ignored
                            let cSharpDefault = 
                                match arg.type_ with
                                | BT.BoolOptional
                                | BT.Bool -> "false"
                                | BT.DoubleOptional
                                | BT.Double -> "0.0"
                                | BT.Int 
                                | BT.IntOptional -> "-1"
                                | TensorOrScalar _ -> "IntPtr.Zero"
                                | BT.ScalarType 
                                | BT.ScalarTypeOptional -> "ScalarType.Float"
                                | BT.DoubleOptional
                                | _ -> "todo"
                            yield sprintf "%s.HasValue" argName
                            yield sprintf "%s%s.GetValueOrDefault(%s)" conversion argName cSharpDefault
                        elif arg.isNullable then 
                            yield conversion + argName
                        else 
                            if arg.type_.ReqPinning 
                            then yield "(IntPtr) p" + argName
                            else yield conversion + argName 
                        if arg.type_.IsArray then
                            yield sprintf "%s.Length" argName
            if schema.HasOptions then yield! [|"(sbyte) dtype"; "device"; "requiresGrad"|]
        |] 

    let checkForErrors = "Torch.CheckForErrors();"

    let emptyDefaults = 
        [| 
            for arg in args do
                let name = arg.name |> toCSharpName
                match arg.ParameterNullable, arg.Nullableable with
                | false,true-> ()
                | false,false -> yield sprintf "_ = %s ?? throw new ArgumentNullException(nameof(%s));" name name
                | true,true -> () // Nullable<T> are handled elsewhere
                | true,false -> 
                    yield
                        match arg.type_ with
                        | BT.TensorVector
                        | BT.TensorList -> 
                            sprintf "%s ??= new TorchTensor[0];" name
                        | TensorOrScalar _ -> 
                            sprintf "IntPtr %sPtr = %s is null ? IntPtr.Zero : %s.Handle;" name name name
                        | BT.IntList -> sprintf "%s ??= new long[0];" name 
                        | _ -> sprintf "// todo %A" arg.type_
        |]

    let nativeCall = 
        [| yield sprintf "%s%s(" THSTensor schema.FunctionName; yield! cArgs |> multiLineParams |> indent|]
        |> compressLines |> addFinalSemiColon

    [|
        match schema.Return with 
        | RT.TensorTuple _ | RT.ManyTensor ->
            yield "IntPtr[] ptrArray;"
            yield! 
                [|yield! nativeCall; yield checkForErrors ; yield "ptrArray = pa.Array;"|]
                |> func "using (var pa = new PinnedArray<IntPtr>())"
            match schema.Return with
            | RT.TensorTuple n -> 
                yield!
                    [|
                        yield "return ("
                        yield! 
                            [|for i in 0 .. n - 1 -> sprintf "new TorchTensor(ptrArray[%i])" i|]
                            |> multiLineParams |> addFinalSemiColon
                    |] |> compressLines
            | RT.ManyTensor -> 
                yield "return ptrArray.Select(x => new TorchTensor(x)).ToArray();"
            | _ -> failwith "err"
        | RT.SingleTensor -> 
            yield! prefixFirstLine  "var res = " nativeCall
            yield checkForErrors
            yield "return new TorchTensor(res);"
        | RT.SingleScalar -> 
            failwith "todo"
        | RT.Single BT.Bool 
        | RT.Single BT.Int
        | RT.Single BT.Double
            -> yield! prefixFirstLine "return "  nativeCall
        | RT.Single BT.ScalarType 
            -> yield! prefixFirstLine "return (ScalarType) " nativeCall
        | _ -> failwith "todo"
    |] 
    |> fun xs -> 
        if anyPinning then 
            args 
            |> Array.filter (fun x -> x.type_.ReqPinning)
            |> Array.groupBy (fun x -> x.type_)
            |> Array.map (fun (_,ys) -> 
                let fixedType = 
                    match ys.[0].type_ with
                    | BT.IntList -> "long"
                    | BT.TensorVector
                    | BT.TensorList -> "IntPtr"
                    | _ -> "todo"

                sprintf "%s* %s" fixedType
                    (ys |> Array.map (fun y -> 
                        let yName = y.name |> toCSharpName
                        let yName' = match y.type_ with | BT.TensorVector | BT.TensorList -> yName + ".Select(x => x.Handle).ToArray()" | _ -> yName
                        sprintf "p%s = %s" yName yName') |> String.concat ", "))
            |> fun ys -> nestedFixed ys xs
        else xs
    |> Array.append emptyDefaults
    |> fun xs -> if isUnsafe then unsafe xs else xs
    |> func firstLine 

    //|> Array.prepend [|""|]
    //|> indent |> indent |> String.concat System.Environment.NewLine


// Array.append [|1|] [|2|]
// Array.prepend [|1|] [|2|]

//let topk = getSchema("topk","") //|> genImport(false)
//let conv2d = getSchema("conv2d","") //|> genImport(false)


// TODO mixed return type
//schema |> genCSharp
//schema.returns
//schema.args

//schema 
//topk |> genCSharp
//conv2d |> genCSharp

//let schema = getSchema("_batch_norm_impl_index","")

// TODO double

//let toDoSchemas = 
//    [|
//        for x in schemas  ->
//            try
//                (x |> genCSharp |> ignore)
//                None
//            with | _ -> Some(x)
//    |] |> Array.choose id
//
//toDoSchemas |> Array.map (fun x -> x.name,x.overloadName)

// filter "from_file", "to", "item"


//[|("convolution_backward_overrideable", ""); ("from_file", "");
//    ("fbgemm_linear_quantize_weight", ""); ("qscheme", ""); ("to", "device");
//    ("item", ""); ("_local_scalar_dense", ""); ("set_quantizer_", "")|]

//schemas 
//|> Array.find (fun x -> x.name = "upsample_linear1d_out")
//|> genCSharp

// 0, convolution_backward_overrideable, Bool3
// 1, from_file BoolOptional
// fbgemm_linear_quantize_weight, mixed Tuple return
// q_scale
// qscheme QScheme type

//toDoSchemas.[0].name

//toDoSchemas.[3].name
//toDoSchemas.[3] |> genCSharp
//
////toDoSchemas.[3].Return
//toDoSchemas.[3] |> genCSharp
//
//toDoSchemas.[4].name
//toDoSchemas.[4] |> genCSharp
//toDoSchemas.[4] 
//schemas |> Array.find (fun x -> x.name = "from_file")
//
//getSchema("from_file","")
//
// "from_file","" // options
//getSchema(,"")

//toDoSchemas 
//|> Array.filter (fun x -> x.args |> Array.exists (fun x -> match x.type_ with | BT.TensorOptionsAnd | BT.TensorOptions -> true | _ -> false))
//
//toDoSchemas 
//|> Array.filter (fun x -> x.args |> Array.exists (fun x -> match x.type_ with | BT.TensorOptionsAnd | BT.TensorOptions -> true | _ -> false))
//|> Array.length

//|> Array.filter (fun x -> x.StartsWith("upsample"))

// 8 failing Returns

//[|("_batch_norm_impl_index", ""); ("fbgemm_linear_quantize_weight", "");
//    ("qscheme", ""); ("result_type", "Tensor"); ("result_type", "Scalar");
//    ("result_type", "Scalar_Tensor"); ("result_type", "Scalar_Scalar");
//    ("promote_types", "")|]
//let failingReturns = toDoSchemas |> Array.filter (fun x -> try x.Return |> ignore; false with | _ -> true) //|> Array.map (fun x -> x.name,x.overloadName)


//|> Array.map (fun x -> x.name)
//|> Array.filter (fun x -> x.StartsWith("upsample") |> not)
//|> Array.length
