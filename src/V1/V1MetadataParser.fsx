#I @".\bin\Debug\"
#r @"YamlDotNet.dll"
#load "SchemaParser.fsx"

open YamlDotNet.RepresentationModel
open System.IO

//type Dispatch = 
//    | CPU
//    | CUDA
//    | MkldnnCPU
//    | QuantizedCPU
//    | QuantizedCUDA
//    | SparseCPU
//    | SparseCUDA
//    static member TryParse(x: string) =
//        match x with
//        | "CPU" -> Some CPU 
//        | "CUDA" -> Some CUDA 
//        | "MkldnnCPU" -> Some MkldnnCPU
//        | "QuantizedCPU" -> Some QuantizedCPU
//        | "QuantizedCUDA" -> Some QuantizedCUDA
//        | "SparseCPU" -> Some SparseCPU
//        | "SparseCUDA" -> Some SparseCUDA
//        | _ -> None

//type Attribute = 
//    | Dispatch of (Dispatch*string)[]
//    | CategoryOverrideFactory 
//    | DeviceGuardFalse
//    | ManualKernelRegistrationTrue
//    | NamedGuardFalse
//    | PythonModuleNN
//    | RequiresTensorTrue
//    | SupportsNamedTensorTrue
//    | UseC10DispatcherFull
//    | Variants of function_ : bool * method_ : bool
    //| Func of func : string * MetadataAST[]

//type Func =  {
//    firstName : string
//    secondName : string
//    inputs : ParamType[]
//    outputs : ParamType[]
//    attributes : Attribute[]
//}

//let loadSchemas(path:string) = 
//    let ys = YamlStream()
//    use sr = new StringReader(File.ReadAllText(path))
//    ys.Load(sr)
//    let doc = ys.Documents.[0]
//    [|
//        let rootNode = (doc.RootNode :?> YamlSequenceNode)
//        let funcs = rootNode.Children
//        for func in funcs do
//            let func = (func :?> YamlMappingNode)
//            let funcValue = ((func.Children |> Seq.find (fun x -> (x.Key :?> YamlScalarNode).ToString() = "func")).Value :?> YamlScalarNode).ToString()
//
//            let attributes = 
//                [|
//                    for kv in func.Children do 
//                        match (kv.Key :?> YamlScalarNode).ToString() with
//                        | "dispatch" ->
//                            yield Attribute.Dispatch([|for kv in (kv.Value :?> YamlMappingNode).Children -> 
//                                                        Dispatch.TryParse(kv.Key.ToString()) |> Option.get, 
//                                                        kv.Value.ToString()|])
//                        | "func" -> ()
//                        | x -> 
//                            yield 
//                                match x,(kv.Value :?> YamlScalarNode).ToString() with
//                                | "use_c10_dispatcher", "full"-> UseC10DispatcherFull
//                                | "variants", "function" -> Variants(true,false)
//                                | "variants", "function, method" 
//                                | "variants", "method, function" -> Variants(true,true)
//                                | "variants", "method" -> Variants(false,true)
//                                | "manual_kernel_registration", "True" -> ManualKernelRegistrationTrue
//                                | "supports_named_tensor", "True" -> SupportsNamedTensorTrue
//                                | "device_guard", "False" -> DeviceGuardFalse
//                                | "requires_tensor", "True" -> RequiresTensorTrue
//                                | "python_module", "nn" -> PythonModuleNN
//                                | "category_override", "factory" -> CategoryOverrideFactory
//                                | "named_guard", "False" -> NamedGuardFalse
//                                | key,value -> 
//                                    failwithf "Error, unexpected attribute %s or value %s" key value
//                    |]
//            failwith "todo"
//    |] 
//
