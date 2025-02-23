using NN;

class Program
{    
    public static float[] td = {
        0, 0, 0, 
        1, 0, 1,
        0, 1, 1,
        1, 1, 0,
    }; 
    public class Xor
    {
        public Mat a0;
        public Mat w1,b1,a1;
        public Mat w2,b2,a2;

        public Xor()
        {
            a0 = new Mat(1,2);

            w1 = new Mat(2,2);
            b1 = new Mat(1,2);
            a1 = new Mat(1,2);
            a2 = new Mat(1,1);
            w2 = new Mat(2,1);
            b2 = new Mat(1,1);

            w2.Rand(-0.5f, 0.5f); b2.Rand(-0.5f, 0.5f);
            w1.Rand(-0.5f, 0.5f); b1.Rand(-0.5f, 0.5f);
        }
    }

    public static float cost(Xor model, Mat ti, Mat to)
    {
        float c = 0.0f;
        int n = to.Rows;
        for(int i = 0; i < n; ++i)
        {
            var x = ti.GetRow(i);
            var y = to.GetRow(i);
            var save = model.a0;
            model.a0 = x;
            ForwardXor(model);
            for(int j = 0; j < model.a2.Cols; ++j)
            {
                float d = model.a2[0,j] - y[0,j];
                c += d*d;
            }
            model.a0 = save;
        }
        return c;
    }

    public static void ForwardXor(Xor model)
    {
        Mat.Mult(model.a1, model.a0, model.w1);
        Mat.Add(model.a1, model.b1);
        Mat.Sigmoid(model.a1);

        Mat.Mult(model.a2, model.a1, model.w2);
        Mat.Add(model.a2, model.b2);
        Mat.Sigmoid(model.a2);
    }
    public static void FiniteDiff(Xor m, Xor g, float eps, Mat ti, Mat to)
    {
        float temp;

        float c = cost(m, ti, to);
        for(var i = 0; i < m.w1.Rows; ++i)
        {
            for(var j = 0; j < m.w1.Cols; ++j)
            {
                temp = m.w1[i,j];
                m.w1[i,j] +=eps;
                g.w1[i,j] = (cost(m,ti,to) - c) / eps;
                m.w1[i,j] = temp;
            }
        }

        for(var i = 0; i < m.b1.Rows; ++i)
        {
            for(var j = 0; j < m.b1.Cols; ++j)
            {
                temp = m.b1[i,j];
                m.b1[i,j] +=eps;
                g.b1[i,j] = (cost(m,ti,to) - c) / eps;
                m.b1[i,j] = temp;
            }
        }
        for(var i = 0; i < m.w2.Rows; ++i)
        {
            for(var j = 0; j < m.w2.Cols; ++j)
            {
                temp = m.w2[i,j];
                m.w2[i,j] +=eps;
                g.w2[i,j] = (cost(m,ti,to) - c) / eps;
                m.w2[i,j] = temp;
            }
        }

        for(var i = 0; i < m.b2.Rows; ++i)
        {
            for(var j = 0; j < m.b2.Cols; ++j)
            {
                temp = m.b2[i,j];
                m.b2[i,j] +=eps;
                g.b2[i,j] = (cost(m,ti,to) - c) / eps;
                m.b2[i,j] = temp;
            }
        }
    }

    public static void Learn(Xor m, Xor g, float rate)
    {
        for(var i = 0; i < m.w1.Rows; ++i)
            for(var j = 0; j < m.w1.Cols; ++j)
                m.w1[i,j] -= g.w1[i,j] * rate;

        for(var i = 0; i < m.b1.Rows; ++i)
            for(var j = 0; j < m.b1.Cols; ++j)
                m.b1[i,j] -= g.b1[i,j] * rate;

        for(var i = 0; i < m.w2.Rows; ++i)
            for(var j = 0; j < m.w2.Cols; ++j)
                m.w2[i,j] -= g.w2[i,j] * rate;

        for(var i = 0; i < m.b2.Rows; ++i)
            for(var j = 0; j < m.b2.Cols; ++j)
                m.b2[i,j] -= g.b2[i,j] * rate;

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
        
        var m = new Xor();
        var g = new Xor();
        
        float eps = 1e-1f;
        float rate = 1e-1f; 
        for(int i = 0; i < 10000; ++i)
        {
            FiniteDiff(m, g, eps, ti, to);
            Learn(m, g, rate);
        }

        for(int i = 0; i < 2; ++i)
        {
            for(int j = 0; j < 2; ++j)
            {
                m.a0[0,0] = i;
                m.a0[0,1] = j;
                ForwardXor(m);
                Console.WriteLine($"{i} ^ {j} = {m.a2[0,0]}");
            }
        }
    }
}
