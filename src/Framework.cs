namespace NN;

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
}
public class Func
{
    private static Random rnd = new Random(Environment.TickCount);
    public static float RandF() => rnd.NextSingle();
    public static float Sigmoidf(float x) =>  1.0f / (MathF.Exp(-x) + 1.0f);
}
