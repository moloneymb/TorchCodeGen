namespace TargetModuleCpp
module Linear = 
    let hearder = """    EXPORT_API(NNModule) THSNN_Linear_ctor(const int64_t input_size, const int64_t output_size, const bool with_bias, NNAnyModule* outAsAnyModule);
    EXPORT_API(Tensor)   THSNN_Linear_forward(const NNModule module, const Tensor tensor);
    EXPORT_API(Tensor)   THSNN_Linear_bias(const NNModule module);
    EXPORT_API(void)     THSNN_Linear_set_bias(const NNModule module, const Tensor tensor);
    EXPORT_API(Tensor)   THSNN_Linear_weight(const NNModule module);
    EXPORT_API(void)     THSNN_Linear_set_weight(const NNModule module, const Tensor tensor);"""

    let cpp = """    NNModule THSNN_Linear_ctor(const int64_t input_size, const int64_t output_size, const bool with_bias, NNAnyModule* outAsAnyModule)
    {
        CATCH_RETURN_NNModule(
            auto opts = torch::nn::LinearOptions(input_size, output_size);
            opts = opts.bias(with_bias);

            auto mod = std::make_shared<torch::nn::LinearImpl>(opts);

            // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
            // a Module can only be boxed to AnyModule at the point its static type is known).
            if (outAsAnyModule != NULL)
            {
                auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<torch::nn::LinearImpl>(*mod));
                *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
            }
            res = new std::shared_ptr<torch::nn::Module>(mod);
        );
    }

    Tensor THSNN_Linear_forward(const NNModule module, const Tensor tensor)
    {
        CATCH_TENSOR((*module)->as<torch::nn::Linear>()->forward(*tensor));
    }

    Tensor THSNN_Linear_bias(const NNModule module)
    {
        CATCH_TENSOR((*module)->as<torch::nn::Linear>()->bias);
    }

    void THSNN_Linear_set_bias(const NNModule module, const Tensor bias)
    {
        CATCH(
            (*module)->as<torch::nn::Linear>()->bias = *bias;
        )
    }

    Tensor THSNN_Linear_weight(const NNModule module)
    {
        CATCH_TENSOR((*module)->as<torch::nn::Linear>()->weight);
    }

    void THSNN_Linear_set_weight(const NNModule module, const Tensor weight)
    {
        CATCH(
            (*module)->as<torch::nn::Linear>()->weight = *weight;
        )
    }"""

    let csharp = """// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class Linear : Module
    {
        internal Linear (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        public new static Linear Load (String modelPath)
        {
            var res = Module.Load (modelPath);
            Torch.CheckForErrors ();
            return new Linear (res.handle.DangerousGetHandle(), IntPtr.Zero);
        }

        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Linear_forward (Module.HType module, IntPtr tensor);

        public TorchTensor Forward (TorchTensor tensor)
        {
            var res = THSNN_Linear_forward (handle, tensor.Handle);
            Torch.CheckForErrors ();
            return new TorchTensor (res);
        }
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Linear_bias (Module.HType module);
        [DllImport ("LibTorchSharp")]
        extern static void THSNN_Linear_set_bias (Module.HType module, IntPtr tensor);

        public TorchTensor? Bias {
            get {
                var res = THSNN_Linear_bias (handle);
                Torch.CheckForErrors ();
                return ((res == IntPtr.Zero) ? null : (TorchTensor ?)new TorchTensor (res));
            }
            set {
                THSNN_Linear_set_bias (handle, (value.HasValue ? value.Value.Handle : IntPtr.Zero));
                Torch.CheckForErrors ();
            }
        }
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Linear_weight (Module.HType module);
        [DllImport ("LibTorchSharp")]
        extern static void THSNN_Linear_set_weight (Module.HType module, IntPtr tensor);

        public TorchTensor Weight {
            get {
                var res = THSNN_Linear_weight (handle);
                Torch.CheckForErrors ();
                return new TorchTensor (res);
            }
            set {
                THSNN_Linear_set_weight (handle, value.Handle);
                Torch.CheckForErrors ();
            }
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Linear_ctor (long input_size, long output_size, bool with_bias, out IntPtr pBoxedModule);

        static public Linear Linear (long inputSize, long outputSize, bool hasBias = true)
        {
            var res = THSNN_Linear_ctor (inputSize, outputSize, hasBias, out var boxedHandle);
            Torch.CheckForErrors ();
            return new Linear (res, boxedHandle);
        }
    }
    public static partial class Functions
    {
        static public TorchTensor Linear (TorchTensor x, long inputSize, long outputSize, bool hasBias = true)
        {
            using (var d = Modules.Linear (inputSize, outputSize, hasBias))
            {
                return d.Forward (x);
            }
        }
    }
}"""

