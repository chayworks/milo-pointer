// On-phone benchmark of the Milo Pointer ONNX model with ONNX Runtime (CPU), run via app_process (no APK needed).
// Reads bench_inputs.txt (one step per line: space-separated token ids), prints per-step latency and the argmax action type.
// Usage on device: CLASSPATH=bench.dex app_process -Djava.library.path=/data/local/tmp/milo /data/local/tmp PointerBench model.onnx inputs.txt threads
import ai.onnxruntime.*;
import java.io.*;
import java.nio.LongBuffer;
import java.util.*;

public class PointerBench {
    public static void main(String[] a) throws Exception {
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions so = new OrtSession.SessionOptions();
        so.setIntraOpNumThreads(Integer.parseInt(a[2]));
        so.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        if (a.length > 3 && a[3].equals("noarena")) { so.setCPUArenaAllocator(false); so.setMemoryPatternOptimization(false); }
        if (a.length > 3 && a[3].equals("xnnpack")) { so.addXnnpack(Collections.singletonMap("intra_op_num_threads", a[2])); so.setIntraOpNumThreads(1); }
        long t0 = System.nanoTime();
        OrtSession s = env.createSession(a[0], so);
        System.out.println("load_ms " + (System.nanoTime() - t0) / 1e6);
        List<long[]> X = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(a[1]))) {
            String l; while ((l = br.readLine()) != null) { String[] p = l.trim().split(" "); long[] v = new long[p.length]; for (int i = 0; i < p.length; i++) v[i] = Long.parseLong(p[i]); X.add(v); }
        }
        int k = 0;
        for (long[] ids : X) {
            OnnxTensor t = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), new long[]{1, ids.length});
            long t1 = System.nanoTime();
            try (OrtSession.Result r = s.run(Collections.singletonMap("ids", t))) {
                double ms = (System.nanoTime() - t1) / 1e6;
                float[] ty = (float[]) r.get(0).getValue(); float[] el = (float[]) r.get(1).getValue();
                int best = 0; for (int i = 1; i < ty.length; i++) if (ty[i] > ty[best]) best = i;
                System.out.println("step " + (k++) + " tokens " + ids.length + " ms " + String.format("%.1f", ms) + " type " + best);
            }
            t.close();
        }
        Runtime rt = Runtime.getRuntime();
        System.out.println("java_heap_mb " + (rt.totalMemory() - rt.freeMemory()) / 1e6);
    }
}
