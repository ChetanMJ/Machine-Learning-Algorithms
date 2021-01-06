
cd C:\Users\cheth\OneDrive\Documents\MachineLearning\Assignment5

C:\Python27\python.exe neuralnet.py smallTrain.csv smallTest.csv model1train_out.labels model1test_out.labels model1metrics_out.txt 2 4 2 0.1

C:\Python27\python.exe neuralnet.py smalltrain.csv smalltest.csv model1train_out.labels model1test_out.labels model1metrics_out.txt 2 4 1 0.1

tar -cvf neuralnet.tar neuralnet.py


C:\Python27\python.exe neuralnet_matmul.py smalltrain.csv smalltest.csv model1train_out.labels model1test_out.labels model1metrics_out.txt 2 4 1 0.1


C:\Python27\python.exe neuralnet.py largeTrain.csv largeTest.csv model1train_out.labels model1test_out.labels model1metrics_out.txt 2 4 2 0.1

C:\Python27\python.exe neuralnet.py mediumTrain.csv mediumTest.csv model1train_out.labels model1test_out.labels model1metrics_out.txt 2 4 2 0.1



    ex = 1e-8
    Alpha = np.array([[1,1,2,-3,0,1,-3], [1,3,1,2,1,0,2], [1,2,2,2,2,2,1], [1,1,0,2,1,-2,2]])
    Beta = np.array([[(1),1,2,-2,1],[1,1,-1,1,2],[1,3,1,-1,1]])
    X = np.array([1,1,1,0,0,1,1])
    Y = np.array([0,1,0])
    
    
    
    ##Beta = np.array([[(1 - ex),0,0,0,0],[0,0,0,1,1],[0,0,0,0,0]])
    NN_Forward = NNForward(X, Y, Alpha, Beta)
    print(NN_Forward.getJ())
    print(NN_Forward.getYCap())

python neuralnet.py smallTrain.csv smallTest.csv model1train_out.labels model1test_out.labels model1metrics_out.txt 2 4 1 0.1

python neuralnet.py largeTrain.csv largeTest.csv model1train_out.labels model1test_out.labels model1metrics_out.txt 2 4 1 0.1
