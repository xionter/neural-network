using NN;

class Program
{    
    public static float[] td = {
        0, 0, 0, 
        1, 0, 1,
        0, 1, 1,
        1, 1, 0,
    }; 

    public static void Main()
    {
        var tdMat = new Mat(4, 3, td);

        var ti = tdMat.GetSubCols(0, 1);
        var to = tdMat.GetCol(2);

        var m = Neural.Model.InitXor();
        var g = Neural.Model.InitXor();
        float eps = 1e-2f;
        float rate = 1f;
        for(int i = 0; i < 10000; ++i)
        {
            Neural.FiniteDiff(m, g, eps, ti, to);
            Neural.Learn(m, g, rate);
            Console.WriteLine(Neural.ModelCost(m, ti, to));
        }

        for(int i = 0; i < 2; ++i)
        {
            for(int j = 0; j < 2; ++j)
            {
                m.Arch[0].a[0,0] = i;
                m.Arch[0].a[0,1] = j;
                Neural.ForwardModel(m);
                Console.WriteLine($"{i} ^ {j} = {m.Arch[^1].a[0,0]}");
            }
        }
    }
}
