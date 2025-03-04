using Neural_Network;

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
        var m = NN.Model.InitXor();
        var g = NN.Model.InitXor();
        float rate = 1f;
        for(int i = 0; i < 100* 1000; ++i)
        {
            NN.BackProp(m, g, ti, to);
            NN.Learn(m, g, rate);
            Console.WriteLine($"{i}: {NN.ModelCost(m, ti, to)}");
        }

        for(int i = 0; i < 2; ++i)
        {
            for(int j = 0; j < 2; ++j)
            {
                m.As[0][0,0] = i;
                m.As[0][0,1] = j;
                NN.ForwardModel(m);
                Console.WriteLine($"{i} ^ {j} = {m.As[^1][0,0]}");
            }
        }
    }
}
