#load "WPFEventLoop.fsx"
#I @".\bin\Debug\"
#r @"DiffPlex.dll"
#r @"DiffPlex.WPF.dll"

open System.Windows.Threading
open WPFEventLoop
open System
open System.Windows
open System.Windows.Controls

/// TODO / NOTE WPFEventLoop may not be needed in this form

WPFEventLoop.Install()
open DiffPlex

// From Utilities.WPF
module Dispatcher =

    let ui = lazy (System.Windows.Threading.DispatcherSynchronizationContext(System.Windows.Application.Current.Dispatcher))

    let switchToDispatcher() : Async<unit> =
        if System.Windows.Application.Current.Dispatcher.CheckAccess() then async.Return()
        else Async.FromContinuations(fun (scont,_,_) -> do System.Windows.Application.Current.Dispatcher.BeginInvoke(System.Action< >(fun () -> scont())) |> ignore)
        
    let runOnMainThread(f) =
            async { do! switchToDispatcher()
                    return f() }
             |> Async.RunSynchronously

    let runOnUIThread(f) =
            async {
                let current = System.Threading.SynchronizationContext.Current    
                do! Async.SwitchToContext  (ui.Force())
                let x = f()
                do! Async.SwitchToContext current 
                return x
            } |> Async.RunSynchronously

    
    let invoke f =
        if System.Windows.Application.Current <> null then
            let dispatcher = System.Windows.Application.Current.Dispatcher
            if dispatcher.CheckAccess() then f()
            else dispatcher.BeginInvoke(Action(fun () -> f())) |> ignore
        else System.Diagnostics.Debug.WriteLine
                "WARN: Attempt to dispatch made with no running application."

    let invokeAsync f =
        let dispatcher = System.Windows.Application.Current.Dispatcher
        dispatcher.BeginInvoke(Action(fun () -> f())) |> ignore

    let invokeSync f =
        let dispatcher = System.Windows.Application.Current.Dispatcher
        if dispatcher.CheckAccess() then f()
        else dispatcher.Invoke(Func<_>(fun () -> f() |> box<'a>))
             |> unbox<'a>


let diffViewer = 
    lazy 
        DiffPlex.Wpf.Controls.DiffViewer(
            IgnoreWhiteSpace = false, 
            FontFamily = Windows.Media.FontFamily("Courier New"))
        

let diffWindow = 
    lazy 
        let window = new Window(Title = "DiffViewer", Width = 400., Height = 300.)
        window.Closing.Add(fun (x:ComponentModel.CancelEventArgs)  -> window.Hide(); printfn "Hiding DiffViewer"; x.Cancel <- true)
        diffViewer.Force().ShowInline()
        window.Content <- diffViewer.Force()
        window

let showDiff (oldText:string) (newText: string) = 
    Dispatcher.invoke (fun () -> 
        let win = diffWindow.Force()
        let viewer = diffViewer.Force()
        win.Show()
        win.Topmost <- false
        win.Topmost <- true
        viewer.NewText <- newText
        viewer.OldText <- oldText
    )


