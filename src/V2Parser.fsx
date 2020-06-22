//#load @"C:\EE\Git\TorchCodeGen\src\TorchMetadataParser.fsx"
#I @".\bin\Debug\"
#r @"YamlDotNet.dll"
#load @"CodeGenCommon.fsx"
open System
open System.IO
open YamlDotNet.RepresentationModel
open CodeGenCommon

// method_of -> namespace

[<RequireQualifiedAccess>]
type BT = 
    | Bool | Bool2 | Bool3 | Bool4
    | BoolOptional
    | ConstQuantizerPtr
    | ConstTensor
    | Device
    | Dimname
    | DimnameList
    | DimnameListOptional
    | Double
    | DoubleOptional
    | Generator
    | Int
    | IntList
    | IntOptional
    | MemoryFormat
    | MemoryFormatOptional
    | Scalar
    | ScalarOptional
    | ScalarType
    | ScalarTypeOptional
    | Storage
    | String
    | Tensor
    | TensorAnd
    | TensorList
    | TensorVector
    | TensorOptionsAnd
    | TensorOptions
    | QScheme
    override this.ToString() = 
        match this with
        | BT.Bool -> "bool"
        | BT.Bool2 -> "std::array<bool,2>"
        | BT.Bool3 -> "std::array<bool,3>"
        | BT.Bool4 -> "std::array<bool,4>"
        | BT.BoolOptional -> "c10::optional<bool>"
        | BT.ConstQuantizerPtr -> "ConstQuantizerPtr"
        | BT.ConstTensor -> "const Tensor &"
        | BT.Device -> "Device"
        | BT.Dimname -> "Dimname"
        | BT.DimnameList -> "DimnameList"
        | BT.DimnameListOptional -> "c10::optional<DimnameList>"
        | BT.Double -> "double"
        | BT.DoubleOptional -> "c10::optional<double>"
        | BT.Generator -> "Generator *"
        | BT.Int -> "int64_t"
        | BT.IntList -> "IntArrayRef"
        | BT.IntOptional -> "c10::optional<int64_t>"
        | BT.MemoryFormat -> "MemoryFormat"
        | BT.MemoryFormatOptional -> "c10::optional<MemoryFormat>"
        | BT.Scalar -> "Scalar"
        | BT.ScalarOptional -> "c10::optional<Scalar>"
        | BT.ScalarType -> "ScalarType"
        | BT.ScalarTypeOptional -> "c10::optional<ScalarType>"
        | BT.Storage -> "Storage"
        | BT.String ->  "std::string"
        | BT.Tensor -> "Tensor"
        | BT.TensorAnd -> "Tensor &"
        | BT.TensorList -> "TensorList"
        | BT.TensorVector -> "std::vector<Tensor>"
        | BT.TensorOptionsAnd -> "const TensorOptions &"
        | BT.TensorOptions -> "TensorOptions"
        | BT.QScheme -> "QScheme"
    static member Parse(str: string) = 
        match str with
        | "bool" -> BT.Bool
        | "std::array<bool,2>" -> BT.Bool2
        | "std::array<bool,3>" -> BT.Bool3
        | "std::array<bool,4>" -> BT.Bool4
        | "c10::optional<bool>" -> BT.BoolOptional
        | "ConstQuantizerPtr" -> BT.ConstQuantizerPtr
        | "const Tensor &" -> BT.ConstTensor
        | "Device" -> BT.Device
        | "Dimname" -> BT.Dimname
        | "DimnameList" -> BT.DimnameList
        | "c10::optional<DimnameList>" -> BT.DimnameListOptional
        | "double" -> BT.Double
        | "c10::optional<double>" -> BT.DoubleOptional
        | "Generator *" -> BT.Generator
        | "int64_t" -> BT.Int
        | "IntArrayRef" -> BT.IntList
        | "c10::optional<int64_t>" -> BT.IntOptional
        | "MemoryFormat" -> BT.MemoryFormat
        | "c10::optional<MemoryFormat>" -> BT.MemoryFormatOptional
        | "Scalar" -> BT.Scalar
        | "c10::optional<Scalar>" -> BT.ScalarOptional
        | "ScalarType" -> BT.ScalarType
        | "c10::optional<ScalarType>" -> BT.ScalarTypeOptional
        | "Storage" -> BT.Storage
        | "std::string" -> BT.String 
        | "Tensor" -> BT.Tensor
        | "Tensor &" -> BT.TensorAnd
        | "TensorList" -> BT.TensorList
        | "std::vector<Tensor>" -> BT.TensorVector
        | "const TensorOptions &" -> BT.TensorOptionsAnd
        | "TensorOptions" -> BT.TensorOptions
        | "QScheme" -> BT.QScheme
        | _ -> failwithf "BT Parse error %s" str
    member this.Metadata = // isArray, isOptional
        match this with
        | BT.Bool -> false,false
        | BT.Bool2 
        | BT.Bool3 
        | BT.Bool4 -> true,false// is it??
        | BT.BoolOptional -> false,true
        | BT.ConstQuantizerPtr -> false,false
        | BT.ConstTensor 
        | BT.Device 
        | BT.Dimname -> false,false
        | BT.DimnameList -> true,false
        | BT.DimnameListOptional -> true,true
        | BT.Double -> false,false
        | BT.DoubleOptional -> false,true
        | BT.Generator -> false,false
        | BT.Int -> false,false
        | BT.IntList -> true,true
        | BT.IntOptional -> false,true
        | BT.MemoryFormat -> false,false
        | BT.MemoryFormatOptional -> false,true
        | BT.Scalar -> false,false
        | BT.ScalarOptional -> false,true
        | BT.ScalarType -> false,false
        | BT.ScalarTypeOptional -> false,true
        | BT.Storage -> false,false
        | BT.String ->  false,false
        | BT.Tensor -> false,false
        | BT.TensorAnd -> false,false
        | BT.TensorList -> true,false
        | BT.TensorVector -> true,false
        | BT.TensorOptionsAnd -> false,false
        | BT.TensorOptions -> false,false
        | BT.QScheme -> false,false
    member this.IsArray = fst this.Metadata
    member this.IsOptional = snd this.Metadata

let (|Tensor|_|) (x:BT) = 
    match x with
    | BT.Tensor
    | BT.TensorAnd
    | BT.TensorList
    //| BT.TensorOptions
    //| BT.TensorOptionsAnd
    | BT.TensorVector
    | BT.ConstTensor -> Some(x)
    | _ -> None

let (|TensorOrScalar|_|) (x:BT) = 
    match x with
    | Tensor _ 
    | BT.Scalar
    | BT.ScalarOptional
    | BT.TensorVector
    | BT.ConstTensor -> Some(x)
    | _ -> None

// Keys [|"annotation"; "dynamic_type"; "is_nullable"; "name"; "type"; "default";
//    "kwarg_only"; "allocate"; "output"; "size"; "field_name"|]
type Arg = {
    name            : string
    annotation      : (char*bool) option
    isNullable     : bool
    defaultValue    : string option
    type_           : BT
    dynamicType     : BT
} 

type Return = {
    dynamicType : BT
    type_ : BT
    fieldName : string option
    name : string
} 

type Schema = {
    name : string
    operatorName : string
    overloadName : string
    args : Arg[]
    returns : Return[]
    depricated : bool
    methodOfTensor : bool option
    methodOfNamespace : bool option
} with member this.ntensors =  
        if this.returns 
            |> Array.exists (fun x -> match x.dynamicType with | BT.Tensor | BT.TensorAnd | BT.ConstTensor -> false | _ -> true) 
            |> not then
            Some(Some(this.returns.Length))
        else 
            match this.returns with
            | [|x|] -> 
                match x.dynamicType with 
                | BT.TensorVector | BT.TensorList -> Some(None)
                | _ -> None
            | _ -> None

let printSchemaSimple(schema: Schema) = 
    let pArgs(xs:Arg[]) : string = 
        [|for x in xs -> 
           sprintf  "%s%s%s" 
                (string x.type_)
                // TODO figure out option
                (match x.name with | "" -> "" | x ->" " + x)
                (match x.defaultValue with | None -> "" | Some(x) -> "=" + x)
        |]
        |> asTuple
    let pReturns(xs:Return[]) = 
        [|for x in xs -> 
           sprintf  "%s%s" 
                (string x.type_)
                // TODO figure out options
                (match x.name with | "" -> "" | x ->" " + x)
                //(match x.defaultValue with | None -> "" | Some(x) -> "=" + x)
        |]
        |> asTuple
    // NOTE: This ignores attributes for now
    printfn "%s.%s%s -> %s" 
        schema.name
        schema.overloadName
        (pArgs(schema.args))
        (pReturns(schema.returns))


let schemas() : Schema[] = 
    let path = @"C:\EE\Git\TorchCodeGen\src\Declarations-v1.5.0.yaml.txt"
    let ys = YamlStream()
    use sr = new StringReader(File.ReadAllText(path))
    ys.Load(sr)
    let doc = ys.Documents.[0]
    let rootNode = (doc.RootNode :?> YamlSequenceNode)
    let returns = YamlScalarNode("returns")
    let arguments = YamlScalarNode("arguments")
    let name = YamlScalarNode("name")
    let operatorName = YamlScalarNode("operator_name")
    let overloadName = YamlScalarNode("overload_name")
    let deprecated = YamlScalarNode("deprecated")
    let method_of = YamlScalarNode("method_of")
    let tensor = YamlScalarNode("Tensor")
    let namespace_ = YamlScalarNode("namespace")

//    for op in rootNode.Children |> Seq.cast<YamlMappingNode> do 
    [|
         for op in rootNode.Children |> Seq.cast<YamlMappingNode> do 
            let args = 
                if op.Children.Keys.Contains(arguments) then
                    op.Children.Item(arguments) :?> YamlSequenceNode
                    |> Seq.cast<YamlMappingNode>
                    |> Seq.map (fun (arg: YamlMappingNode) -> 
                        let getV(name:string) = (arg.Item(YamlScalarNode(name)) :?> YamlScalarNode).Value 
                        let annotation = 
                            match getV "annotation" with
                            | "null" -> None
                            | "a" -> Some('a', false)
                            | "a!" -> Some('a', true)
                            | "b!" -> Some('b', true)
                            | "c!" -> Some('c', true)
                            | _ -> failwith "err"
                        {
                            name            = getV "name"
                            annotation      = annotation
                            isNullable      = match getV "is_nullable" with | "true" -> true | "false" -> false | _ -> failwith "err"
                            defaultValue    = if arg.Children.Keys.Contains(YamlScalarNode("default")) then Some(getV "default") else None
                            type_           = BT.Parse(getV "type")
                            dynamicType     = BT.Parse(getV "dynamic_type")
                        })
                    |> Seq.toArray
                else [||]
            let returns = 
                if op.Children.Keys.Contains(returns) then
                    op.Children.Item(returns) :?> YamlSequenceNode
                    |> Seq.cast<YamlMappingNode>
                    |> Seq.map (fun ret -> 
                        let getV(name:string) = (ret.Item(YamlScalarNode(name)) :?> YamlScalarNode).Value 
                        {
                            name         = getV "name"
                            dynamicType  = BT.Parse(getV "dynamic_type")
                            type_        = BT.Parse(getV "type")
                            fieldName    = if ret.Children.Keys.Contains(YamlScalarNode("field_name")) then Some(getV "field_name") else None
                        })
                    |> Seq.toArray
                else [||]

            let mo = 
                if op.Children.Keys.Contains(method_of) then
                    let x = (op.Children.Item(method_of) :?> YamlSequenceNode)
                    Some(x.Children.Contains(tensor), x.Children.Contains(namespace_))
                else None
            {
                name = (op.Children.Item(name) :?> YamlScalarNode).Value
                operatorName = (op.Children.Item(operatorName) :?> YamlScalarNode).Value
                overloadName = (op.Children.Item(overloadName) :?> YamlScalarNode).Value
                args = args
                returns = returns
                depricated = (match (op.Children.Item(deprecated) :?> YamlScalarNode).Value with | "true" -> true | "false" -> false | _ -> failwith "err")
                methodOfTensor = mo |> Option.map fst
                methodOfNamespace = mo |> Option.map snd
            }
    |]


//let schemas2 = schemas()
//let x = schemas2.[0]


//YamlScalarNode("default") :> YamlNode |> fun x -> arguments.[0].Children.Keys |> Seq.exists ((=) x)
//YamlScalarNode("type") :> YamlNode |> fun x -> arguments.[0].Children.Keys |> Seq.exists ((=) x)
//
////arguments |> Array.collect (fun x -> x.Children.Keys |> Seq.toArray |> Array.map (fun x -> (x :?> YamlScalarNode).Value)) |> Array.distinct
//arguments.[0]
//
////(((rootNode.Children.[0] :?> YamlMappingNode).Children.Item(YamlScalarNode("arguments")) :?> YamlSequenceNode).Children.[0].Item(YamlScalarNode("dynamic_type")) :?> YamlScalarNode).Value
//
////for x in ((rootNode.Children.[0] :?> YamlMappingNode).Children.Item(YamlScalarNode("arguments")) :?> YamlSequenceNode).Children do
////    (x :?> YamlMappingNode).["annotation"]
//
////    let doc = ys.Documents.[0]
////    [|
////        let rootNode = (doc.RootNode :?> YamlSequenceNode)
////        let funcs = rootNode.Children
////        for func in funcs do
////            let func = (func :?> YamlMappingNode)
////            let funcValue = ((func.Children |> Seq.find (fun x -> (x.Key :?> YamlScalarNode).ToString() = "func")).Value :?> YamlScalarNode).ToString()
////
////            let attributes = 
////                [|
////                    for kv in func.Children do 
////                        match (kv.Key :?> YamlScalarNode).ToString() with
////                        | "dispatch" ->
////                            yield Attribute.Dispatch([|for kv in (kv.Value :?> YamlMappingNode).Children -> 
////                                                        Dispatch.TryParse(kv.Key.ToString()) |> Option.get, 
////                                                        kv.Value.ToString()|])
////                        | "func" -> ()
////                        | x -> 
////                            yield 
////                                match x,(kv.Value :?> YamlScalarNode).ToString() with
////                                | "use_c10_dispatcher", "full"-> UseC10DispatcherFull
////                                | "variants", "function" -> Variants(true,false)
////                                | "variants", "function, method" 
////                                | "variants", "method, function" -> Variants(true,true)
////                                | "variants", "method" -> Variants(false,true)
////                                | "manual_kernel_registration", "True" -> ManualKernelRegistrationTrue
////                                | "supports_named_tensor", "True" -> SupportsNamedTensorTrue
////                                | "device_guard", "False" -> DeviceGuardFalse
////                                | "requires_tensor", "True" -> RequiresTensorTrue
////                                | "python_module", "nn" -> PythonModuleNN
////                                | "category_override", "factory" -> CategoryOverrideFactory
////                                | "named_guard", "False" -> NamedGuardFalse
////                                | key,value -> 
////                                    failwithf "Error, unexpected attribute %s or value %s" key value
////                    |]
////            match run funcParser funcValue with
////            | ParserResult.Success(((names,inputs),outputs),_,_) -> 
////                let f (((baseType: BaseType,modifiers: Modifier list),name: string option),default_: string option) :  ParamType =
////                    {
////                        baseType  = baseType
////                        array = modifiers |> List.choose (function | Modifier.Array(x) -> Some(x) | _ -> None) |> List.tryHead 
////                        alpha = modifiers |> List.choose (function | Modifier.Alpha(x,y) -> Some(x,y) | _ -> None) |> List.tryHead
////                        optional = modifiers |> List.exists ((=) Modifier.Optional)
////                        name = name |> Option.defaultValue ""
////                        defaultValue = default_
////                    } 
////                    
////                yield {
////                    firstName = match names with | [x] | [x;_] -> x | _ -> failwith "err"
////                    secondName = match names with | [_] -> "" | [_;x] -> x | _ -> failwith "err"
////                    inputs = inputs |> List.map f |> List.toArray
////                    outputs = outputs |> List.map f |> List.toArray
////                    attributes = attributes
////                }
////            | ParserResult.Failure(_) as x -> failwithf "%A" x
////            
