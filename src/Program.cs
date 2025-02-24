using NN;

class Program
{    
    public static float[] td = {
        0, 0, 0, 
        1, 0, 1,
        0, 1, 1,
        1, 1, 0,
    }; 

    public class Layer
    {
        public Mat a{get;set;}
        public Mat b{get;set;}
        public Mat w{get;set;} 

        public Layer(Mat b, Mat w)
        {
            this.b = b; this.w = w;
            this.a = new Mat(b.Rows, b.Cols);
        }
    }

    public class Model
    {
        public Mat Input{get;set;}
        public Layer[] Arch{get;set;}

        public Model(Mat Input, int n, Layer[] layers)
        {
            this.Input = Input;
            Arch = layers;
        }

        public static Model InitModelXor()
        {
            Mat a0 = new Mat(1,2); //Input

            Mat w1 = new Mat(2,2);
            Mat b1 = new Mat(1,2);
            Mat a1 = new Mat(1,2);

            Mat w2 = new Mat(2,1);
            Mat a2 = new Mat(1,1);
            Mat b2 = new Mat(1,1);

            w1.Rand(-0.1f, 0.1f); b1.Rand(-0.1f, 0.1f);
            w2.Rand(-0.1f, 0.1f); b2.Rand(-0.1f, 0.1f);
            var l1 = new Layer(b1, w1);
            var l2 = new Layer(b2, w2);
            return new Model(a0, 2, new Layer[]{l1, l2});
        }
    }

    public static float ModelCost(Model m, Mat ti, Mat to)
    {
        if(ti.Rows != to.Rows) throw new InvalidOperationException("Input and output rows dont match");
        if(m.Arch[^1].a.Cols != to.Cols) throw new InvalidOperationException("model output cols and data output cols dont match");
        if(m.Input.Cols != ti.Cols) throw new InvalidOperationException("model Input cols and data Input cols dont match");

        float c = 0.0f;
        int n = to.Rows;
        for(int i = 0; i < n; ++i)
        {
            var x = ti.GetRow(i);
            var y = to.GetRow(i);
            m.Input = x;
            ForwardModel(m);
            for(int j = 0; j < to.Cols; ++j)
            {
                float d = m.Arch[^1].a[0,j] - y[0,j];
                c += d*d;
            }
        }
        return c;
    }

    public static void ForwardModel(Model m)
    {
        Mat.Mult(m.Arch[0].a, m.Input, m.Arch[0].w);
        Mat.Add(m.Arch[0].a, m.Arch[0].b);
        Mat.Sigmoid(m.Arch[0].a);

        for(int i = 1; i < m.Arch.Length; ++i)
        {
            Mat.Mult(m.Arch[i].a, m.Arch[i-1].a, m.Arch[i].w);
            Mat.Add(m.Arch[i].a, m.Arch[i].b);
            Mat.Sigmoid(m.Arch[i].a);
        }
    }

    public static void FiniteDiff(Model m, Model g, float eps, Mat ti, Mat to)
    {
        float temp;

        float c = ModelCost(m, ti, to);
        for(int i = 0; i < m.Arch.Length; ++i)
        {
            for(int j = 0; j < m.Arch[i].w.Rows; ++j)
            {
                for(int k = 0; k < m.Arch[i].w.Cols; ++k)
                {
                    temp = m.Arch[i].w[j, k];
                    m.Arch[i].w[j, k] += eps;
                    g.Arch[i].w[j, k] = (ModelCost(m,ti,to) - c) / eps;
                    m.Arch[i].w[j, k] = temp;
                }
            }
            for(int j = 0; j < m.Arch[i].b.Rows; ++j)
            {
                for(int k = 0; k < m.Arch[i].b.Cols; ++k)
                {
                    temp = m.Arch[i].b[j, k];
                    m.Arch[i].b[j, k] += eps;
                    g.Arch[i].b[j, k] = (ModelCost(m,ti,to) - c) / eps;
                    m.Arch[i].b[j, k] = temp;
                }
            }
        }
    }

    public static void Learn(Model m, Model g, float rate)
    {
        for(int i = 0; i < m.Arch.Length; ++i)
        {
            for(int j = 0; j < m.Arch[i].w.Rows; ++j)
            {
                for(int k = 0; k < m.Arch[i].w.Cols; ++k)
                {
                    m.Arch[i].w[j, k] -= g.Arch[i].w[j, k] * rate;
                }
            }
            for(int j = 0; j < m.Arch[i].b.Rows; ++j)
            {
                for(int k = 0; k < m.Arch[i].b.Cols; ++k)
                {
                    m.Arch[i].b[j, k] -= g.Arch[i].b[j, k] * rate;
                }
            }
        }
    }
    public static void Main()
    {
        var numCols = 3; 
        var numRows = td.Length / numCols;

        var inp = new float[numRows * 2];
        var outp = new float[numRows];

        for(int i = 0; i < numRows; ++i)
        {
            inp[i*2] = td[i*3];
            inp[i*2 + 1] = td[i*3 + 1];
            outp[i] = td[i*3 + 2];
        }
        var ti = new Mat(numRows, 2, inp);
        var to = new Mat(numRows, 1, outp);
        var m = Model.InitModelXor();
        var g = Model.InitModelXor();

        float eps = 1e-2f;
        float rate = 0.5f;
        for(int i = 0; i < 10000; ++i)
        {
            FiniteDiff(m, g, eps, ti, to);
            Learn(m, g, rate);
        }

        for(int i = 0; i < 2; ++i)
        {
            for(int j = 0; j < 2; ++j)
            {
                m.Input[0,0] = i;
                m.Input[0,1] = j;
                ForwardModel(m);
                Console.WriteLine($"{i} ^ {j} = {m.Arch[^1].a[0,0]}");
            }
        }
    }
}
