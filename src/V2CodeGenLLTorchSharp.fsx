// TODO add these Scalars back in [|"S2";"S";"S_"|] 

#load "V2Parser.fsx"
#load @"DiffPlex.fsx"
#load "Clippy.fsx"
#load "TargetModuleCPP.fsx"
#load "Common.fsx"
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

let schemas = V2Parser.schemas()

let tryGetFunctionName(line: string) = 
        match line.IndexOf("THSTensor_") with
        | -1 -> None
        | n -> 
            // Take until
            let sub = line.Substring(n + 10)
            Some(sub.Substring(0,sub.IndexOf("(")))

let filteredFunctions = 
    let ignoreList = 
        set [
            "cpu"; "cuda";"deviceType"; "dispose"; "grad"; 
            "new";"newLong" // this is from_blob
            "isSparse"; //"requires_grad" 
            "sum1"
            "sparse"; "stride"; "strides"
            "to_type"; "type"
            "ndimension"; "size"
            ]
    File.ReadAllLines(THSTensorh).[6..] 
    |> Array.choose tryGetFunctionName
    |> Array.filter (fun x -> 
        [|"S2";"S";"S_"|] |> Array.exists (fun y -> x.EndsWith(y)) |> not)
    |> Array.filter (fun x -> x.EndsWith("T") |> not)
    |> Array.filter (fun x -> (x.StartsWith("set") || x.StartsWith("get")) |> not)
    |> Array.filter (fun x -> (x.EndsWith("Scalar")) |> not)
    |> Array.filter (fun x -> ignoreList.Contains(x) |> not)

let matches = 
    let nameMapping = 
        [|
            ("empty",("new_empty","")) 
            // Not sure about this "names" is nullable but options are not
            (*("empty",("empty","names")) *) 
            ("norm",("norm","ScalarOpt_dim"))
            // Not sure about this
            ("requires_grad",("requires_grad_",""))
            ("scatter",("scatter","src"))
            ("transpose",("transpose","int"))
            ("unbind",("unbind","int"))
            ("max",("max","dim"))
            ("squeeze",("squeeze","dim"))
        |] |> Map.ofArray
    filteredFunctions 
    |> Array.choose (fun x -> 
        match nameMapping.TryFind(x) with
        | Some(name,overload) -> 
            schemas 
            |> Array.tryFind (fun x -> x.name = name && x.overloadName = overload)
        | None -> 
            schemas 
            |> Array.tryFind (fun y ->  
                y.name = x && 
                (y.overloadName = "" ||
                 y.overloadName = "Tensor" || 
                 y.overloadName = "input" || // e.g. conv_transpose2d
                 y.overloadName = "Tensor_Tensor" // i.e. pow(T,T)
                )))


let argName(x:string) = 
    match x with
    | "N" -> "n"
    | "output_size" -> "sizes"
    | _ -> x


let TensorAllocator = "Tensor* (*allocator)(size_t length)"

type Variant = 
    {
        defaultGuard : bool
        dropArgs : Map<(string*string),string[]>
    } 
    static member Default = {defaultGuard = false; dropArgs = Map.empty}

let torchSharp = {defaultGuard = true; dropArgs = [||] |> Map.ofArray}

let variant = torchSharp // Variant.Default


let argCSignature(x:Arg) = 
    let name = argName x.name
    match x.type_ with
    | BT.TensorOptions
    | BT.TensorOptionsAnd when x.name = "options" -> "const int8_t scalar_type, const char* device, const bool requires_grad"
    | BT.TensorOptions
    | BT.TensorOptionsAnd -> sprintf "int %s_kind, int %s_device" name name

    | BT.Bool2 | BT.Bool3 | BT.Bool4 -> failwithf "todo %A" x
    | BT.IntList -> sprintf "const int64_t* %s, const int %s_length" name name
    //| BT.TensorList when x.isNullable -> sprintf "int %s_kind, int %s_device" name name
    | BT.TensorList (* when not x.isNullable *) -> sprintf "const Tensor* %s_data, const int %s_length" name name
    | BT.DimnameList -> failwithf "todo %A" x
    | BT.String -> Printf.sprintf "char* %s_ptr, int %s_len" name name

        //| BaseType.Strig -> 
        //    sprintf "char* %s_ptr, int %s_len" name name
    | _ when x.type_.IsArray -> failwithf "err %A" x
    | _ ->
        match x.type_ with
        | BT.Bool -> "const bool" // when is this bool or int??
        | BT.Int
        | BT.IntOptional-> "const int64_t"
        | BT.Double -> "double"
        | BT.Tensor -> "Tensor"
        | BT.TensorAnd -> "const Tensor"
        | BT.ConstTensor -> "const Tensor"
        | BT.ScalarType 
        | BT.ScalarTypeOptional -> "int"
        //| BT.Device -> "int"
        | BT.Device -> "const int8_t" // unsure about this...
        | BT.Scalar -> "const Scalar"
        | BT.ScalarOptional -> "const Scalar" 
        | BT.MemoryFormat 
        | BT.MemoryFormatOptional -> "const MemoryFormat"
        | _ -> (string x.type_) //failwithf "err %A" x
        |> fun x -> sprintf "%s %s" x name 
    |> fun y -> if x.isNullable || (not(variant.defaultGuard) && x.defaultValue.IsSome) then sprintf "const bool with_%s, %s" x.name y else y

let genHeader(schema: Schema) : string = 
    printfn "%s" schema.name
    //let name = schema.name |> underscoreToCamel |> capitalizeFirst

    let hasAllocator, returnType = 
        match schema.returns with
        | [||] -> false, "void"
        | [|x|] -> 
            match x.dynamicType with
            | BT.Tensor -> false, "Tensor"
            | BT.Scalar -> false, "Scalar"
            | BT.Bool -> false, "int"
            | BT.TensorVector
            | BT.TensorList -> true, "void" 
            | _ -> failwith "todo"
        | [|x;y|] when x.dynamicType = BT.Tensor && y.dynamicType = BT.Tensor -> true, "void"
            // This is the common two_tuple
        | _ -> failwith "todo"

    let overloadedName = 
        match schema.overloadName with
        | "Tensor" -> schema.name
        | "Scalar" -> schema.name + "S" // TODO swap underscore "_S"
        | "" -> schema.name
        | _ -> printfn "%s unhandled" schema.overloadName ; schema.name

    let args = 
        schema.args |> Array.map argCSignature 
        |> fun xs -> if hasAllocator then [|yield xs.[0]; yield TensorAllocator; yield! xs.[1..]|] else xs
        |> String.concat ", "
    sprintf "EXPORT_API(%s) THSTensor_%s(%s);" returnType overloadedName args


let f(x:Schema) = true // x.name.Contains("add")

matches
|> Array.filter f
|> Array.sortBy (fun x -> x.name)
|> fun xs -> 
    let names = xs |> Array.map (fun x -> x.name) |> Set.ofArray
    let newText = xs |> Array.map genHeader |> String.concat Environment.NewLine
    let oldText = 
        File.ReadAllLines(THSTensorh).[6..] 
        |> Array.choose (fun x -> 
            tryGetFunctionName(x) 
            |> Option.bind (fun y -> if names.Contains(y) then Some(y,x) else None ))
        |> Array.sortBy fst |> Array.map snd |> String.concat Environment.NewLine
    newText |> DiffPlex.showDiff oldText


schemas |> Array.find (fun x -> x.name = "max" && x.overloadName = "dim")

//schemas |> Array.filter (fun x -> x.name = "addbmm")

schemas 
|> Array.filter (fun x -> x.name = "unbind") 
|> Array.map (fun x -> x.name, x.overloadName)

DiffPlex.diffViewer.Value.IgnoreWhiteSpace <- true
DiffPlex.diffViewer.Value.IgnoreCase <- true
DiffPlex.diffViewer.Value.ShowSideBySide()

matches |> Array.find (fun x -> x.name = "abs")


matches 
|> Array.choose (fun x -> x.returns |> Array.tryHead)
|> Array.map (fun x -> x.dynamicType)
|> Array.countBy id

