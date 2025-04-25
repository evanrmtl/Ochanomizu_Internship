# Linear Regression

### Result linear regression
*********

Actual value: 130.0
Predicted value: 176.64
**********
Actual value: 195.0
Predicted value: 197.73000000000002
**********
Actual value: 218.0
Predicted value: 249.34500000000003
**********
Actual value: 166.0
Predicted value: 193.84500000000003
**********
Actual value: 163.0
Predicted value: 214.38
**********
Actual value: 155.0
Predicted value: 164.985
**********
Actual value: 204.0
Predicted value: 178.86
**********
Actual value: 270.0
Predicted value: 203.28000000000003
**********
Actual value: 205.0
Predicted value: 164.43
**********
Actual value: 127.0
Predicted value: 137.79000000000002
**********
Actual value: 260.0
Predicted value: 211.60500000000002
**********
Actual value: 249.0
Predicted value: 238.245
**********
Actual value: 251.0
Predicted value: 236.025
**********
Actual value: 158.0
Predicted value: 158.325
**********
Actual value: 167.0
Predicted value: 190.51500000000001
**********

```haskell

import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Functional (matmul, mul, add, transpose2D, sumAll)
import Torch.Functional.Internal (mulScalar, square, size)

targetValues :: Tensor
targetValues = asTensor ([130, 195, 218, 166, 163, 155, 204, 270, 205, 127, 260, 249, 251, 158, 167] :: [Double])

featureValues :: Tensor
featureValues = asTensor ([148, 186, 279, 179, 216, 127, 152, 196, 126, 78, 211, 259, 255, 115, 173] :: [Double])

linear :: 
    (Tensor, Tensor) -> -- ^ parameters (slope, intercept)
    Tensor ->           -- ^ input features
    Tensor              -- ^ predictions
linear (slope, intercept) input = add (mul input slope) intercept



main :: IO()
main = do
    let initialSlope = asTensor ([0.555] :: [Double])
    let initialIntercept = asTensor ([94.5] :: [Double])

    let initialPredictions = linear (initialSlope, initialIntercept) featureValues
    let actualValuesList = asValue targetValues :: [Double]


    mapM_ (\(actual, predicted) -> do
      putStrLn $ "Actual value: " ++ show actual
      putStrLn $ "Predicted value: " ++ show predicted
      putStrLn "**********")
      (zip actualValuesList (asValue initialPredictions :: [Double]))
```



### Result Cost function

cost = 558.7061566666666

```haskell
cost ::
    Tensor -> -- ^ ground truth values
    Tensor -> -- ^ predicted values
    Tensor    -- ^ loss (scalar)
cost groundTruth predictions = 
  let diff = (predictions - groundTruth)
      squared = square diff
      sumSquared = sumAll squared
      n = fromIntegral (size squared 0)
  in (sumSquared / (2 * n)) 
  ```

### Caluculate new a & b

```haskell

calculateNewA :: 
     Tensor -> -- ^ input features
     Tensor -> -- ^ target values
     Tensor -> -- ^ current slope
     Tensor -> -- ^ current intercept
     Float ->  -- ^ learning rate
     Tensor    -- ^ updated slope
calculateNewA features targets currentSlope currentIntercept learningRate = 
  let numSamples = fromIntegral (size features 0)
      predictions = linear (currentSlope, currentIntercept) features
      errors = predictions - targets
      gradient = sumAll (errors * features) / numSamples
  in currentSlope - mulScalar gradient learningRate

calculateNewB :: 
     Tensor -> -- ^ input features
     Tensor -> -- ^ target values
     Tensor -> -- ^ current slope
     Tensor -> -- ^ current intercept
     Float ->  -- ^ learning rate
     Tensor    -- ^ updated intercept
calculateNewB features targets currentSlope currentIntercept learningRate = 
  let numSamples = fromIntegral (size features 0)
      predictions = linear (currentSlope, currentIntercept) features
      errors = predictions - targets
      gradient = sumAll errors / numSamples
  in currentIntercept - mulScalar gradient learningRate
```

I did add some arguments because i didn't knew how i could do it with only 2 tensors.


And here is the result with those:

Epoch: 1, Loss: 134549.04789478387, a: 3.2756535032046186, b: 99.96589066752834 <br/>
Epoch: 2, Loss: 11834.479864134266, a: 1.3250701548717596, b: 99.95598966402024 <br/>
Epoch: 3, Loss: 1508.6492862967593, a: 0.7592493364843749, b: 99.9531109584088<br/>
Epoch: 4, Loss: 639.7811536521749, a: 0.5951173335461561, b: 99.95226926526622<br/>
Epoch: 5, Loss: 566.6701468258294, a: 0.547506323173536, b: 99.95201846415316<br/>
Epoch: 6, Loss: 560.5182092524772, a: 0.5336954615386238, b: 99.95193906768912<br/>
Epoch: 7, Loss: 560.0005497895813, a: 0.5296892714013659, b: 99.95190939191365<br/>
Epoch: 8, Loss: 559.9569872665171, a: 0.5285271984282512, b: 99.95189413901582<br/>
Epoch: 9, Loss: 559.9533176816474, a: 0.5281901405966332, b: 99.95188306988564<br/>
Epoch: 10, Loss: 559.9530048923821, a: 0.528092401310735, b: 99.951873214385<br/>
Epoch: 20, Loss: 559.9529327555657, a: 0.5280529199590887, b: 99.9517794165389<br/>
Epoch: 30, Loss: 559.9528889556738, a: 0.5280533944149391, b: 99.95168582294582<br/>
Epoch: 40, Loss: 559.9528451573099, a: 0.5280538690310608, b: 99.95159223098582<br/>
Epoch: 50, Loss: 559.9528013604734, a: 0.528054343638906, b: 99.95149864065802<br/>
Epoch: 60, Loss: 559.9527575651645, a: 0.5280548182384743, b: 99.9514050519624<br/>
Epoch: 70, Loss: 559.9527137713831, a: 0.5280552928297659, b: 99.9513114648989<br/>
Epoch: 80, Loss: 559.9526699791292, a: 0.5280557674127809, b: 99.9512178794675<br/>
Epoch: 90, Loss: 559.9526261884026, a: 0.5280562419875192, b: 99.95112429566822<br/>
Epoch: 100, Loss: 559.9525823992036, a: 0.5280567165539813, b: 99.95103071350098<br/>
Target slope = 0.555, Final slope = 0.5280567165539813<br/>
Target intercept = 94.6, Final intercept = 99.95103071350098<br/>

### Graduate .5



```haskell
import LinearRegression (linear, cost, calculateNewA, calculateNewB, train)
import Data (loadData, dataToTensor, targetToTensor, cgpaToTensor)
import Torch.Tensor (indexSelect', Tensor, asTensor, size, asValue)

initialA :: Tensor
initialA = asTensor ([1] :: [Double])

initialB :: Tensor
initialB = asTensor ([1] :: [Double])

epochsPerIteration :: Int
epochsPerIteration = 100

learningRate :: Float
learningRate = 1e-2

trainMultipleIterations ::
    Int ->                   -- Number of iterations left
    (Tensor, Tensor) ->      -- Training data (features, targets)
    (Tensor, Tensor) ->      -- Evaluation data (features, targets)
    (Tensor, Tensor) ->      -- Current model parameters (a, b)
    IO (Tensor, Tensor)      -- Final model parameters (a, b)
trainMultipleIterations 0 _ _ finalParams = return finalParams
trainMultipleIterations iterationsLeft (trainFeatures, trainTargets) (evalFeatures, evalTargets) (paramA, paramB) = do
    (newParamA, newParamB) <- train epochsPerIteration learningRate trainFeatures trainTargets (paramA, paramB)
    
    let predictions = linear (newParamA, newParamB) evalFeatures
        evaluationLoss = cost predictions evalTargets
    
    putStrLn ( "*****************************************************************\n" 
        ++ "Iteration: " ++ show (11 - iterationsLeft)
        ++ ", Loss: " ++ show (asValue evaluationLoss :: Double)
        ++ ", a: " ++ show (asValue newParamA :: Double)
        ++ ", b: " ++ show (asValue newParamB :: Double) ++
        "\n*****************************************************************\n" )
    
    trainMultipleIterations (iterationsLeft - 1) (trainFeatures, trainTargets) (evalFeatures, evalTargets) (newParamA, newParamB)

main :: IO()
main = do
    trainData <- loadData "data/train.csv"
    evalData <- loadData "data/eval.csv"
    validData <- loadData "data/valid.csv"
    
    case (trainData, evalData, validData) of
        (Right trainDataset, Right evalDataset, Right validDataset) -> do
            let trainPair = (cgpaToTensor trainDataset, targetToTensor trainDataset)
                evalPair = (cgpaToTensor evalDataset, targetToTensor evalDataset)
                (validFeatures, validTargets) = (cgpaToTensor validDataset, targetToTensor validDataset)
            finalParams <- trainMultipleIterations 10 trainPair evalPair (initialA, initialB)
            let validPredictions = linear finalParams validFeatures
                validationLoss = cost validPredictions validTargets
            
            putStrLn $ "Validation result: " ++ show validationLoss

        (Left err, _, _) -> putStrLn ("Error in train file: " ++ show err)
        (_, Left err, _) -> putStrLn ("Error in evaluation file: " ++ show err)
        (_, _, Left err) -> putStrLn ("Error in validation file: " ++ show err)

```

BEG
Loss : 1121.3130113957022        a: 5.0     b: 5.0
*****************************************************************
Iteration: 1, Loss: 7.147319741409966e-2, a: -0.41831347736697055, b: 4.346922954212433
*****************************************************************

*****************************************************************
Iteration: 2, Loss: 7.080012836522234e-2, a: -0.41536574626714945, b: 4.321455656814019
*****************************************************************

*****************************************************************
Iteration: 3, Loss: 7.013344726212009e-2, a: -0.412431869859537, b: 4.296108058789427
*****************************************************************

*****************************************************************
Iteration: 4, Loss: 6.947309384139307e-2, a: -0.40951178302540514, b: 4.270879597537131
*****************************************************************

*****************************************************************
Iteration: 5, Loss: 6.881900840647541e-2, a: -0.40660542095209334, b: 4.245769713099913
*****************************************************************

*****************************************************************
Iteration: 6, Loss: 6.81711318223117e-2, a: -0.4037127191315685, b: 4.220777848152426
*****************************************************************

*****************************************************************
Iteration: 7, Loss: 6.752940551008231e-2, a: -0.4008336133589923, b: 4.19590344798881
*****************************************************************

*****************************************************************
Iteration: 8, Loss: 6.68937714419799e-2, a: -0.39796803973129735, b: 4.171145960510394
*****************************************************************

*****************************************************************
Iteration: 9, Loss: 6.626417213603403e-2, a: -0.3951159346457684, b: 4.146504836213438
*****************************************************************

*****************************************************************
Iteration: 10, Loss: 6.564055065098423e-2, a: -0.3922772347986298, b: 4.12197952817693
*****************************************************************

Validation result: 8.163913701713214e-2