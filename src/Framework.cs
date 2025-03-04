namespace Neural_Network;

public class NN
{
    public class Model
    {
        public Mat[] As{get;set;}
        public Mat[] Ws{get;set;}
        public Mat[] Bs{get;set;}
        public int Count{get;set;}

        public Model(int[] arch)
        {
            Count = arch.Length;
            As = new Mat[Count];
            Ws = new Mat[Count - 1];
            Bs = new Mat[Count - 1];

            As[0] = new Mat(1, arch[0]);
            for(int i = 1; i < Count; ++i)
            {
                int wRows = arch[i - 1];
                int wCols = arch[i];
                int bRows = 1;
                int bCols = arch[i];

                Ws[i - 1] = new Mat(wRows, wCols);
                Bs[i - 1] = new Mat(bRows, bCols);
                As[i] = new Mat(bRows, bCols);

                Ws[i - 1].Rand(0.0f, 1.0f);
                Bs[i - 1].Rand(0.0f, 1.0f);
            }
        }

        public static Model InitXor()
        {
            int[] arch = {2, 2, 1};
            return new Model(arch);
        }

        public void Fill(float num)
        {
            for(int i = 0; i < Count - 1; ++i)
            {
                for(int j = 0; j < As[i].Rows;++j)
                    for(int k = 0; k < As[i].Cols;++k)
                        As[i][j, k] = num;

                for(int j = 0; j < Ws[i].Rows;++j)
                    for(int k = 0; k < Ws[i].Cols;++k)
                        Ws[i][j, k] = num;

                for(int j = 0; j < Bs[i].Rows;++j)
                    for(int k = 0; k < Bs[i].Cols;++k)
                        Bs[i][j, k] = num;
            }
            for(int j = 0; j < As[^1].Rows;++j)
                for(int k = 0; k < As[^1].Cols;++k)
                    As[^1][j, k] = num;
        }

        public void Print()
        {
            for(int i = 0; i < Count - 1;++i)
            {
                As[i].Print($"a{i}");
                Ws[i].Print($"w{i + 1}");
                Bs[i].Print($"b{i + 1}");
            }
            As[^1].Print($"a{Count - 1}");
        }
    }

    public static float ModelCost(Model m, Mat ti, Mat to)
    {
        if(ti.Rows != to.Rows) throw new InvalidOperationException("data input and output rows dont match");
        if(m.As[^1].Cols != to.Cols) throw new InvalidOperationException("model output cols and data output cols dont match");
        if(m.As[0].Cols != ti.Cols) throw new InvalidOperationException("model input cols and data input cols dont match");

        float c = 0.0f;
        int n = to.Rows;
        for(int i = 0; i < n; ++i)
        {
            var x = ti.GetRow(i);
            var y = to.GetRow(i);
            m.As[0] = x;
            ForwardModel(m);
            for(int j = 0; j < to.Cols; ++j)
            {
                float d = m.As[^1][0,j] - y[0,j];
                c += d*d;
            }
        }
        return c / n;
    }

    public static void ForwardModel(Model m)
    {
        for(int i = 0; i < m.Count - 1; ++i)
        {
            Mat.Mult(m.As[i+1], m.As[i], m.Ws[i]);
            Mat.Add(m.As[i+1], m.Bs[i]);
            Mat.Sigmoid(m.As[i+1]);
        }
    }
    
    public static void BackProp(Model m, Model g, Mat ti, Mat to)
    {
        if(ti.Rows != to.Rows) throw new InvalidOperationException("data input and output rows dont match");
        if(m.As[^1].Cols != to.Cols) throw new InvalidOperationException("model output cols and data output cols dont match");
        if(m.As[0].Cols != ti.Cols) throw new InvalidOperationException("model input cols and data input cols dont match");

        g.Fill(0.0f);

        int n = ti.Rows;
        for(int i = 0; i < n; ++i)
        {
            var x = ti.GetRow(i);
            var y = to.GetRow(i);
            m.As[0] = x;
            ForwardModel(m);

            for(int j = 0; j < g.Count;++j)
                for(int k = 0; k < g.As[j].Cols;++k)
                    g.As[j][0, k] = 0.0f;
            
            for(int j = 0; j < to.Cols; ++j)
                g.As[^1][0, j] = m.As[^1][0, j] - y[0, j]; 

            for(int l = m.Count - 1; l > 0; --l)
            {
                for(int j = 0; j < m.As[l].Cols; ++j)
                {
                    float da = g.As[l][0, j];
                    float a = m.As[l][0, j];
                    float sigmoid_grad = a * (1 - a);
                    float delta = da * sigmoid_grad;
                    g.Bs[l - 1][0, j] += delta;

                    for(int k = 0; k < m.As[l-1].Cols;++k)
                    {
                        float w = m.Ws[l - 1][k, j];
                        float pa = m.As[l-1][0, k];
                        g.Ws[l - 1][k, j] += delta*pa;
                        g.As[l - 1][0, k] += delta*w;
                    }
                }
            }
        }

        for(int i = 0; i < m.Count - 1; ++i)
            for(int j = 0; j < g.Ws[i].Rows;++j)
                for(int k = 0; k < g.Ws[i].Cols;++k)
                    g.Ws[i][j, k] /= n;

        for(int i = 0; i < m.Count - 1; ++i)
            for(int j = 0; j < g.Bs[i].Rows;++j)
                for(int k = 0; k < g.Bs[i].Cols;++k)
                    g.Bs[i][j, k] /= n;
    }

    public static void FiniteDiff(Model m, Model g, float eps, Mat ti, Mat to)
    {
        if(ti.Rows != to.Rows) throw new InvalidOperationException("data input and output rows dont match");
        if(m.As[^1].Cols != to.Cols) throw new InvalidOperationException("model output cols and data output cols dont match");
        if(m.As[0].Cols != ti.Cols) throw new InvalidOperationException("model input cols and data input cols dont match");

        float temp;

        float c = ModelCost(m, ti, to);
        for(int i = 0; i < m.Count - 1; ++i)
        {
            for(int j = 0; j < m.Ws[i].Rows; ++j)
            {
                for(int k = 0; k < m.Ws[i].Cols; ++k)
                {
                    temp = m.Ws[i][j, k];
                    m.Ws[i][j, k] += eps;
                    g.Ws[i][j, k] = (ModelCost(m,ti,to) - c) / eps;
                    m.Ws[i][j, k] = temp;
                }
            }
            for(int j = 0; j < m.Bs[i].Rows; ++j)
            {
                for(int k = 0; k < m.Bs[i].Cols; ++k)
                {
                    temp = m.Bs[i][j, k];
                    m.Bs[i][j, k] += eps;
                    g.Bs[i][j, k] = (ModelCost(m,ti,to) - c) / eps;
                    m.Bs[i][j, k] = temp;
                }
            }
        }
    }

    public static void Learn(Model m, Model g, float rate)
    {
        for(int i = 0; i < m.Count - 1; ++i)
        {
            for(int j = 0; j < m.Ws[i].Rows; ++j)
            {
                for(int k = 0; k < m.Ws[i].Cols; ++k)
                {
                    m.Ws[i][j, k] -= g.Ws[i][j, k] * rate;
                }
            }
            for(int j = 0; j < m.Bs[i].Rows; ++j)
                for(int k = 0; k < m.Bs[i].Cols; ++k)
                    m.Bs[i][j, k] -= g.Bs[i][j, k] * rate;
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

    public Mat GetCol(int i)
    {
        var col = new float[Rows];
        for(int j = 0; j < Rows; ++j)
            col[j] = Elements[j * Cols + i];

        return new Mat(Rows, 1, col);
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

        for(int y = 0; y < dst.Rows; ++y)
            for(int x = 0; x < dst.Cols; ++x)
                dst[y,x] = 0;

        for(int y = 0; y < a.Rows; ++y)
            for(int x = 0; x < b.Cols; ++x)
                for(int k = 0; k < a.Cols; ++k)
                    dst[y,x] += a[y,k] * b[k, x];
    }

    public void Print(string name)
    {
        Console.WriteLine($"{name} = ");
        Console.WriteLine("[");
        for(int y = 0; y < Rows; ++y)
        {
            for(int x = 0; x < Cols; ++x)
            {
                Console.Write($"{this[y,x]} ");
            }
            Console.WriteLine();
        }
        Console.WriteLine("]");
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
        var rows = endY - startY + 1;
        var cols = endX - startX + 1;
        var data = new float[rows * cols];
        int idx = 0;
        for(int i = startY; i <= endY; ++i)
            for(int j = startX; j <= endX; ++j)
                data[idx++] = Elements[i * Cols + j];

        return new Mat(rows, cols, data);
    }

    public Mat GetSubCols(int start, int end)
    {
        int cols = end - start + 1;
        var data = new float[Rows * cols];
        int idx = 0;
        for(int i = 0; i < Rows; ++i)
            for(int j = start; j <= end; ++j)
                data[idx++] = Elements[i * Cols + j];

        return new Mat(Rows, cols, data);
    }
}

public class Func
{
    private static Random rnd = new Random(Environment.TickCount);
    public static float RandF() => rnd.NextSingle();
    public static float Sigmoidf(float x) =>  1.0f / (MathF.Exp(-x) + 1.0f);
}
