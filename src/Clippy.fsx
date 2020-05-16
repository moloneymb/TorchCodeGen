module Clippy
// MIT License
// https://github.com/kolibridev/clippy/blob/master/Clippy/Clippy.cs

open System
open System.Runtime.InteropServices
open System.Text

module Native = 
    [<DllImport("kernel32.dll")>]
    extern IntPtr GlobalAlloc(uint64 uFlags, UIntPtr dwBytes);
    
    [<DllImport("kernel32.dll")>]
    extern uint64 GetLastError();

    [<DllImport("kernel32.dll")>]
    extern IntPtr LocalFree(IntPtr hMem);

    [<DllImport("kernel32.dll")>]
    extern IntPtr GlobalFree(IntPtr hMem);
    
    [<DllImport("kernel32.dll")>]
    extern IntPtr GlobalLock(IntPtr hMem);

    [<DllImport("kernel32.dll")>]
    extern [<MarshalAs(UnmanagedType.Bool)>] bool GlobalUnlock(IntPtr hMem);

    [<DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)>]
    extern void CopyMemory(IntPtr dest, IntPtr src, uint64 count);

    [<DllImport("user32.dll")>]
    extern [<MarshalAs(UnmanagedType.Bool)>] bool OpenClipboard(IntPtr hWndNewOwner);

    [<DllImport("user32.dll")>]
    extern [<MarshalAs(UnmanagedType.Bool)>] bool CloseClipboard();

    [<DllImport("user32.dll")>]
    extern IntPtr SetClipboardData(uint64 uFormat, IntPtr data);

    type ResultCode = 
        | Success = 0
        | ErrorOpenClipboard = 1
        | ErrorGlobalAlloc = 2
        | ErrorGlobalLock = 3
        | ErrorSetClipboardData = 4
        | ErrorOutOfMemoryException = 5
        | ErrorArgumentOutOfRangeException = 6
        | ErrorException = 7
        | ErrorInvalidArgs = 8
        | ErrorGetLastError = 9

    type TextEncoding = 
        | ASCII
        | UNICODE
        member this.format = match this with | ASCII -> 1UL | UNICODE -> 13UL

    type Result = {ResultCode : ResultCode; LastError : uint64} member this.OK = this.ResultCode = ResultCode.Success


    [<STAThread>]
    let private __PushStringToClipboard(message: string, format: TextEncoding) : Result = 
        if message = null then {ResultCode = ResultCode.ErrorInvalidArgs; LastError = 0UL} else
        if not(OpenClipboard(IntPtr.Zero)) then { ResultCode = ResultCode.ErrorOpenClipboard; LastError = GetLastError() } else
        let sizeOfChar = uint64 sizeof<char>
        let characters = uint64 (message.Length);
        let bytes = (characters + 1UL) * sizeOfChar
        let GMEM_MOVABLE = 0x0002UL;
        let GMEM_ZEROINIT = 0x0040UL;
        let GHND = GMEM_MOVABLE ||| GMEM_ZEROINIT;
        // IMPORTANT: SetClipboardData requires memory that was acquired with GlobalAlloc using GMEM_MOVABLE.
        let mutable hGlobal = GlobalAlloc(GHND, UIntPtr(bytes))
        if hGlobal = IntPtr.Zero then { ResultCode = ResultCode.ErrorGlobalAlloc; LastError = GetLastError() } else
        // IMPORTANT: Marshal.StringToHGlobalUni allocates using LocalAlloc with LMEM_FIXED.
        //            Note that LMEM_FIXED implies that LocalLock / LocalUnlock is not required.
        let source = match format with | ASCII -> Marshal.StringToHGlobalAnsi(message) | UNICODE -> Marshal.StringToHGlobalUni(message)
        let target = GlobalLock(hGlobal)
        if target = IntPtr.Zero then { ResultCode = ResultCode.ErrorGlobalLock; LastError = GetLastError() }
        else
            try
                try
                    CopyMemory(target, source, bytes) 
                    if SetClipboardData(format.format, hGlobal).ToInt64() <> 0L 
                    then
                        // IMPORTANT: SetClipboardData takes ownership of hGlobal upon success.
                        hGlobal <- IntPtr.Zero
                        { ResultCode = ResultCode.Success; LastError = 0UL};
                    else 
                        // Marshal.StringToHGlobalUni actually allocates with LocalAlloc, thus we should theorhetically use LocalFree to free the memory...
                        // ... but Marshal.FreeHGlobal actully uses a corresponding version of LocalFree internally, so this works, even though it doesn't
                        //  behave exactly as expected.
                        { ResultCode = ResultCode.ErrorSetClipboardData; LastError = GetLastError() }
                finally 
                    GlobalUnlock(target) |> ignore
                    Marshal.FreeHGlobal(source);
                    if hGlobal <> IntPtr.Zero then GlobalFree(hGlobal) |> ignore
                    CloseClipboard() |> ignore
            with
            | :? OutOfMemoryException -> {ResultCode = ResultCode.ErrorOutOfMemoryException; LastError = GetLastError() }
            | :? ArgumentOutOfRangeException -> { ResultCode = ResultCode.ErrorArgumentOutOfRangeException; LastError = GetLastError() };
            | _ -> try { ResultCode = ResultCode.ErrorException; LastError = GetLastError() }; with _ -> { ResultCode = ResultCode.ErrorGetLastError; LastError = 0UL };

    [<STAThread>]
    let PushUnicodeStringToClipboard(message: string) = __PushStringToClipboard(message, UNICODE)

    [<STAThread>]
    let PushAnsiStringToClipboard(message : string) = __PushStringToClipboard(message, ASCII);

    [<STAThread>]
    let PushStringToClipboard(message: string) =
        if message <> null && (message = Encoding.ASCII.GetString(Encoding.ASCII.GetBytes(message))) 
        then PushUnicodeStringToClipboard(message)
        else PushAnsiStringToClipboard(message)

let copyToClipboard(x: string) = 
    let result = Native.PushStringToClipboard(x) 
    if not result.OK then failwithf "Error %A" result

