
(*

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
*)

//let fullSet = schemas |> Array.map (fun x -> x.name, x.overloadName) |> Set
//let rustSet = V2Filtered.filtered_schemas() |> Array.map (fun x -> x.name,x.overloadName) |> Set
//let currentSet = ts |> Array.choose id |> Set

// topk.returns
//getSchema("unbind","") |> genImport(false)

//getSchema("topk","")

//currentSet |> Set.toArray |> Array.map (getSchema >> genCpp) |> String.concat (Environment.NewLine + Environment.NewLine) |> Clippy.copyToClipboard

//Set.difference rustSet currentSet |> Set.toArray |> Array.map (fun (x,y) -> sprintf "%s,%s" x y) |> String.concat System.Environment.NewLine |> Clippy.copyToClipboard

//Set.difference fullSet rustSet |> Set.toArray |> Array.map (fun (x,y) -> sprintf "%s,%s" x y) |> String.concat System.Environment.NewLine |> Clippy.copyToClipboard



// Fixed number

//getSchema("rnn_tanh","data")
//getSchema("rnn_tanh","input")

//genHeader (true,true) (getSchema("rnn_tanh","data")) |> Clippy.copyToClipboard
//genHeader (true,true) (getSchema("rnn_tanh","input"))

//errored.[4] |> getSchema 
//errored.[4] |> getSchema |> genHeader (true,false)
//errored.[4]

//errored
//|> Array.map getSchema 
//|> Array.filter (fun x -> x.args |> Array.exists (fun y -> match y.type_ with | BT.Dimname | BT.DimnameList | BT.DimnameListOptional -> true | _ -> false))
//|> Array.length

//let convT2d = schemas |> Array.find (fun x -> x.name = "conv_transpose2d")
//let sigmoid = schemas |> Array.find (fun x -> x.name = "sigmoid")
//let mp = schemas |> Array.find (fun x -> x.name.Contains("max_pool1d_with_indices"))
//let mp = schemas |> Array.find (fun x -> x.name.Contains("split_with_sizes"))

//let randn = schemas |> Array.find (fun x -> x.name.Contains("randint"))
//genCpp(randn)

//schemas |> Array.find (fun x -> x.name = "max_pool")
//genCpp(sigmoid)
//genCpp({sigmoid with methodOfNamespace = None})

//mp.args |> Array.find (fun x -> x.name = "stride")

//sigmoid.args |> Array.map cppToC
//sigmoid.methodOfTensor
//sigmoid.methodOfNamespace
// sigmoid
// maxpool1d_with_indices multi-out allocator fixed
// split_with_sizes multi-out variable
// sum with optional dtype
// subS2
// max


//schemas |> Array.map (fun x -> (x.args |> Array.filter (fun x -> x.defaultValue.IsSome && not x.isNullable) |> Array.length), x.name) |> Array.sortByDescending fst


//let args = schemas |> Array.collect (fun x -> x.args)
//args |> Array.filter (fun x -> x.isNullable) |> Array.map (fun x -> x.dynamicType) |> Set
//args |> Array.filter (fun x -> x.isNullable |> not) |> Array.map (fun x -> x.dynamicType) |> Set

//schemas |> Array.find (fun x -> x.name = "_cnamesudnn_rnn_backward")

//schemas 
//|> Array.find (fun x -> x.name = "rand" && x.overloadName = "names")
//|> genCpp

// schemas |> Array.find (fun x -> x.name = "norm" && x.overloadName = "ScalarOpt_dim_dtype") |> genCpp

//|> fun x -> x.args |> Array.find (fun x -> x.type_.IsArray)

// Args that are not tensors are nullable and don't have default are 
// Things that are very few things that, DimnameListOptional?

//schemas 
//|> Array.map (fun x -> (x.args |> Array.filter (fun x -> x.dynamicType <> BT.Tensor && x.isNullable && x.defaultValue.IsNone) |> Array.length,(x.name,x.overloadName))) 
//|> Array.sortByDescending fst
//|> Array.filter (function | (0,_) -> false | _ -> true)


//schemas |> Array.filter (fun x -> x.name = "norm") |> Array.map (fun x -> x.name, x.overloadName)

//let f(x:Schema) = true // x.name.Contains("add")
//
//DiffPlex.diffViewer.Value.IgnoreWhiteSpace <- true
//DiffPlex.diffViewer.Value.IgnoreCase <- true
//DiffPlex.diffViewer.Value.ShowSideBySide()
//
//matches
//|> Array.filter f
//|> Array.sortBy (fun x -> x.name)
//|> fun xs -> 
//    let names = xs |> Array.map (fun x -> x.name) |> Set.ofArray
//    let newText = 
//        xs 
//        |> Array.map (genHeader true) 
//        |> Array.map (fun x -> x + ";") 
//        |> String.concat Environment.NewLine
//    let oldText = 
//        File.ReadAllLines(THSTensorh).[6..] 
//        |> Array.choose (fun x -> 
//            tryGetFunctionName(x) 
//            |> Option.bind (fun y -> if names.Contains(y) then Some(y,x) else None ))
//        |> Array.sortBy fst |> Array.map snd |> String.concat Environment.NewLine
//    newText |> DiffPlex.showDiff oldText


//let errored = 
//    V2Filtered.filtered_schemas() |> Array.map (fun x -> 
//        match x.overloadName with
//        | "Tensor"
//        | "Scalar"
//        | "" ->
//            (try genCpp(x) |> ignore; None  with | _ -> Some(x.name, x.overloadName))
//        | _ -> None)
//    |> Array.choose id
//
//let ts = 
//    V2Filtered.filtered_schemas() |> Array.map (fun x -> 
//        match x.overloadName with
//        | "Tensor"
//        | "Scalar"
//        | "" ->
//            (try genCpp(x) |> ignore; Some(x.name,x.overloadName) with | _ -> None)
//        | _ -> None)
