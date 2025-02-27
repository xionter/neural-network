namespace NN;

public class Neural
{
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

        public Layer(int wRows, int wCols, int bRows, int bCols)
        {
            w = new Mat(wRows, wCols);
            w.Rand(0.0f, 1.0f);
            b = new Mat(bRows, bCols);
            b.Rand(0.0f, 1.0f);
            a = new Mat(bRows, bCols);
        }

    }

    public class Model
    {
        public Layer[] Arch{get;set;}

        public Model(int[] arch)
        {
            int n = arch.Length;
            Arch = new Layer[n];
            int inputSize = arch[0];

            Arch[0] = new Layer(0, 0, 1, arch[0]);
            for(int i = 1; i < n; ++i)
            {
                int wRows = arch[i - 1];
                int wCols = arch[i];
                int bRows = 1;
                int bCols = arch[i];

                Arch[i] = new Layer(wRows, wCols, bRows, bCols);
            }
        }

        public static Model InitXor()
        {
            int[] arch = {2, 2, 1};
            return new Model(arch);
        }
    }

    public static float ModelCost(Model m, Mat ti, Mat to)
    {
        if(ti.Rows != to.Rows) throw new InvalidOperationException("data input and output rows dont match");
        if(m.Arch[^1].a.Cols != to.Cols) throw new InvalidOperationException("model output cols and data output cols dont match");
        if(m.Arch[0].a.Cols != ti.Cols) throw new InvalidOperationException("model input cols and data input cols dont match");

        float c = 0.0f;
        int n = to.Rows;
        for(int i = 0; i < n; ++i)
        {
            var x = ti.GetRow(i);
            var y = to.GetRow(i);
            m.Arch[0].a = x;
            ForwardModel(m);
            for(int j = 0; j < to.Cols; ++j)
            {
                float d = m.Arch[^1].a[0,j] - y[0,j];
                c += d*d;
            }
        }
        return c / n;
    }

    public static void ForwardModel(Model m)
    {
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

}

public class Mat
{
    public int Rows{get;}
    public int Cols{get;}
    public float[] Elements{get;set;} 

    public Mat(int rows, int cols)
    {
        Rows = rows; Cols = cols; Elements = new float[Rows * Cols];
    }

    public Mat(int rows, int cols, float[] elements)
    {
        Rows = rows; Cols = cols; Elements = elements;
    }

    public float this[int i, int j]
    {
        get => Elements[i * Cols + j];
        set => Elements[i * Cols + j] = value;
    }       //взрыв мозга, лазеры из глаз
    
    public Mat GetRow(int i)
    {
        var row = new float[Cols];
        Array.Copy(Elements, i * Cols, row, 0, Cols);
        return new Mat(1, Cols, row);
    }

    public static void Add(Mat a, Mat b)
    {
        if(a.Rows != b.Rows) throw new InvalidOperationException("Matrix rows dont match");
        if(a.Cols != b.Cols) throw new InvalidOperationException("Matrix cols dont match");
    
        for(int y = 0; y < a.Rows; ++y)
            for(int x = 0; x < a.Cols; ++x)
                a[y,x]+= b[y,x];
    }
    public static void Mult(Mat dst, Mat a, Mat b)
    {
        if(a.Cols != b.Rows) throw new InvalidOperationException("Matrix sizes dont match");
        if(dst.Rows != a.Rows) throw new InvalidOperationException("Matrix sizes dont match");
        if(dst.Cols != b.Cols) throw new InvalidOperationException("Matrix sizes dont match");

        for(int y = 0; y < a.Rows; ++y)
            for(int x = 0; x < b.Cols;++x)
                for(int k = 0; k < a.Cols; ++k)
                    dst[y,x] = 0;

        for(int y = 0; y < a.Rows; ++y)
            for(int x = 0; x < b.Cols;++x)
                for(int k = 0; k < a.Cols; ++k)
                    dst[y,x] += a[y,k] * b[k, x];
    }

    public void Print()
    {
        for(int y = 0; y < Rows; ++y)
        {
            for(int x = 0; x < Cols; ++x)
            {
                Console.Write($"{this[y,x]} ");
            }
            Console.WriteLine();
        }

    }

    public void Rand(float low, float high)
    {
        for(int y = 0; y < Rows; ++y)
            for(int x = 0; x < Cols; ++x)
                this[y,x] = Func.RandF() * (high - low) + low;
    }

    public static void Sigmoid(Mat m)
    {
        for(int y = 0; y < m.Rows; ++y)
            for(int x = 0; x < m.Cols; ++x)
                m[y,x] = Func.Sigmoidf(m[y,x]);
    }

    public Mat GetSubmatrix(int startY, int startX, int endY, int endX)
    {
        var rows = Math.Abs(endY - startY + 1);
        var cols = Math.Abs(endX - startX + 1);
        var data = new float[rows * cols];
        int idx = 0;
        for(int i = startY; i <= endY; ++i)
            for(int j = startX; j <= endX; ++j)
                data[idx++] = Elements[i * Cols + j];

        return new Mat(rows, cols, data);
    }
}

public class Func
{
    private static Random rnd = new Random(Environment.TickCount);
    public static float RandF() => rnd.NextSingle();
    public static float Sigmoidf(float x) =>  1.0f / (MathF.Exp(-x) + 1.0f);
}
