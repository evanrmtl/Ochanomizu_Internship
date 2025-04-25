import LinearRegression (linear, cost, calculateNewA, calculateNewB, train)
import Data (loadData, dataToTensor, targetToTensor, cgpaToTensor)
import Torch.Tensor (indexSelect', Tensor, asTensor, size, asValue)

initialA :: Tensor
initialA = asTensor ([5] :: [Double])

initialB :: Tensor
initialB = asTensor ([5] :: [Double])

epochsPerIteration :: Int
epochsPerIteration = 100

learningRate :: Float
learningRate = 0.01

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
            putStrLn ("BEG\nLoss : " 
                ++ show (asValue (cost (targetToTensor trainDataset) (linear (initialA, initialB) (cgpaToTensor trainDataset))):: Double)
                ++ "        a: " ++ show (asValue initialA :: Double) ++ "     b: " ++ show ( asValue initialB :: Double))
            finalParams <- trainMultipleIterations 10 trainPair evalPair (initialA, initialB)
            let validPredictions = linear finalParams validFeatures
                validationLoss = cost validPredictions validTargets
            
            putStrLn $ "Validation result: " ++ show (asValue validationLoss :: Double)

        (Left err, _, _) -> putStrLn ("Error in train file: " ++ show err)
        (_, Left err, _) -> putStrLn ("Error in evaluation file: " ++ show err)
        (_, _, Left err) -> putStrLn ("Error in validation file: " ++ show err)