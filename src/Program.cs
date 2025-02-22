using NN;

class Program
{    
    public float[] td = {
        0, 0, 0, 
        1, 0, 0,
        0, 1, 0,
        1, 1, 0,
    }; 
    public struct Xor
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
            w1.Rand(0, 1); b1.Rand(0, 1);

            w2 = new Mat(2,1);
            b2 = new Mat(1,1);
            a2 = new Mat(1,1);
            w1.Rand(0, 1); b1.Rand(0, 1);
        }
    }

    public static float cost(Xor model, Mat ti, Mat to)
    {
        float cost = 0.0f;
        int n = to.Rows;
        for(int i = 0; i < n; ++i)
        {
            var x = ti.GetRow(i);
            var y = to.GetRow(i);
            model.a0 = x;
            forward_xor(model);

            for(int j = 0; j < model.a2.Cols; ++j)
            {
                float d = model.a2[i,j] - y[i,j];
                cost += d*d;
            }
        }
        return cost;
    }

    static void forward_xor(Xor model)
    {

        Mat.Mult(model.a1, model.a0, model.w1);
        Mat.Add(model.a1, model.b1);
        Mat.Sigmoid(model.a1);

        Mat.Mult(model.a2, model.a1, model.w2);
        Mat.Add(model.a2, model.b2);
        Mat.Sigmoid(model.a2);

    }
    public static void Main()
    {
        var m = new Xor();

    }
}
