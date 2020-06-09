#load "V2Parser.fsx"
#load "DiffPlex.fsx"
open CodeGenCommon
open V2Parser

let noTensorOptions =
    set [ "zeros_like"; "empty_like"; "full_like"; "ones_like"; "rand_like"; "randint_like"; "randn_like"; ]

let excludedFunctions  = set [
      "multi_margin_loss";
      "multi_margin_loss_out";
      "log_softmax_backward_data";
      "softmax_backward_data";
      "clone";
      "copy_";
      "conv_transpose2d_backward_out";
      "conv_transpose3d_backward_out";
      "slow_conv_transpose2d_backward_out";
      "slow_conv_transpose3d_backward_out";
      "slow_conv3d_backward_out";
      "normal";
      "_cufft_set_plan_cache_max_size";
      "_cufft_clear_plan_cache";
      "backward";
      "set_data";
      "_amp_non_finite_check_and_unscale_";
      "_cummin_helper";
      "_cummax_helper";
      "retain_grad"; ]

let excludedPrefixes = set [ "_thnn_"; "_th_"; "thnn_"; "th_" ]
let excludedSuffixes = set [ "_forward"; "_forward_out" ]

let methods() = 
    let baseFunc : Schema = 
        {name = ""; operatorName = ""; overloadName = ""; args = [||]; returns = [||]; depricated = false; methodOfTensor = Some(true); methodOfNamespace = None} 
    let f (name: string)  (baseType: BT) : Arg = 
        { type_ = baseType;  name = name; defaultValue = None; isNullable = false; annotation = None; dynamicType = baseType }
    [|
        "grad", [||]
        "set_requires_grad", [|f "r" BT.Bool|]
        "toType", [|f "scalar_type" BT.ScalarType|]
        "to", [|f "device" BT.Device|]
    |] |> Array.map (fun (name,inputs) -> 
        {baseFunc with name = name; args = [|yield f "self" BT.Tensor; yield! inputs|]})

let filtered_schemas() = 
    V2Parser.schemas()
    |> Array.filter (fun x -> not x.depricated) 
    |> Array.filter (fun x -> not (x.overloadName = "deprecated"))
    |> Array.filter (fun x -> excludedPrefixes |> Seq.exists (fun y -> x.name.StartsWith(y)) |> not) 
    |> Array.filter (fun x -> excludedSuffixes|> Seq.exists (fun y -> x.name.EndsWith(y)) |> not) 
    |> Array.filter (fun x -> x.overloadName.EndsWith("generator") |> not)
    |> Array.filter (fun x -> x.overloadName.EndsWith("generator_out") |> not)
    |> Array.filter (fun x -> x.overloadName.StartsWith("source_") |> not)
    |> Array.filter (fun x -> excludedFunctions.Contains(x.name) |> not) // 20

//    |> Array.map 
//    //|> Array.groupBy (fun x -> x.name)
//    // TODO 
//    |> Array.collect ( function | (x,[|y|]) -> [|(x,y)|] 
//                                //| (_,ys) -> ys |> Array.map (fun y -> (sprintf "%s_%s" y.firstName y.secondName, y)))
//                                | (_,ys) -> ys |> Array.mapi (fun i y  -> (sprintf "%s%s" y.name (if i = 0 then "" else string i),y)))
//    |> Array.map (fun (x,y) -> x.ToLower(),y)
//    |> Map.ofArray

