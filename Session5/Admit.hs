{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}


import Control.Monad (when)
import Data.List (foldl', intersperse, scanl')
import GHC.Generics
import Torch
import qualified Data.ByteString.Lazy as BL
import qualified Data.Vector as V
import Data.Csv (decodeByName, FromNamedRecord)
import ML.Exp.Chart (drawLearningCurve)
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Functional (mul, add, sub, sigmoid)
import Torch.Functional.Internal (greater_ts)
import Torch.Device (Device)
import Control.Monad (forM, forM_)
import Data (targetToTensor, dataToTensor, loadData)
import Evaluation (precision, recall, accuracy, f1score, microF1score, macroF1score, weightedF1score)


epochsPerIteration :: Int
epochsPerIteration = 200

learningRate :: Float
learningRate = 0.0001

--bidju hijam

data MLPspec = MLPspec 
    {
        feature_counts :: [Int],
        nonlinearitySpec :: Tensor -> Tensor       
    }

data MLP = MLP
    {
        layer :: [Linear],
        nonlinearity :: Tensor -> Tensor
    } deriving (Generic, Parameterized)

instance Randomizable MLPspec MLP where
  sample MLPspec {..} = do
    let layer_sizes = mkLayerSizes feature_counts
    linears <- mapM sample $ map (uncurry LinearSpec) layer_sizes
    return $ MLP {layer = linears, nonlinearity = nonlinearitySpec}
    where
      mkLayerSizes (a : (b : t)) =
        scanl shift (a, b) t
        where
          shift (a, b) c = (b, c)

mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} input = foldl' revApply input $ intersperse nonlinearity $ map linear layer
  where
    revApply x f = f x

train :: MLP -> Tensor -> Tensor -> IO MLP
train model features targets = do
    (model, lossValues) <- foldLoop (model, []) epochsPerIteration $ \(current, losses) i -> do
        let pred = mlp current features
            loss = mseLoss targets pred
            lossValue = asValue loss :: Float
        (newCurrent, _) <- runStep current optimizer loss (asTensor learningRate)
        when (i `mod` 100 == 0) $ do
            putStrLn "------------------------------------------------------------"
            putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show lossValue
        return (newCurrent, losses ++ [lossValue])
    
    drawLearningCurve "curveTrain.png" "Learning Curve" [("Training Loss", lossValues)]
    return model
    where
        optimizer = GD

train' :: MLP -> Tensor -> Tensor -> IO MLP
train' model features targets = do
    (model, lossValues) <- foldLoop (model, []) epochsPerIteration $ \(current, losses) i -> do
        let pred = mlp current features
            loss = mseLoss targets pred
            lossValue = asValue loss :: Float
        (newCurrent, _) <- runStep current optimizer loss (asTensor learningRate)
        when (i `mod` 100 == 0) $ do
            putStrLn "------------------------------------------------------------"
            putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show lossValue
        return (newCurrent, losses ++ [lossValue])
    
    drawLearningCurve "curveEval.png" "Learning Curve" [("Training Loss", lossValues)]
    return model
    where
        optimizer = GD


eval :: MLP -> Tensor -> Tensor -> IO ()
eval model features target = do
    let modelPredic = mlp model features
    let pred = toDType Bool $ greater_ts modelPredic 0.5
    putStrLn $ "Predictions: " ++ show (asValue (toDType Float modelPredic) :: [Float])
    putStrLn $ "Targets: " ++ show (asValue (toDType Float target) :: [Float])

    let tp = sumAll (mul (toDType Float target) (toDType Float pred))
        tn = sumAll (mul (toDType Float (1 - target)) (toDType Float (1 - pred)))
        fp = sumAll (mul (toDType Float (1 - target)) (toDType Float pred))
        fn = sumAll (mul (toDType Float target) (toDType Float (1 - pred)))

    putStrLn $ "True Positives: " ++ show (asValue tp :: Float)
    putStrLn $ "True Negatives: " ++ show (asValue tn :: Float)
    putStrLn $ "False Positives: " ++ show (asValue fp :: Float)
    putStrLn $ "False Negatives: " ++ show (asValue fn :: Float)

    let precisionValue = precision tp fp
        recallValue = recall tp fn
        accuracyValue = accuracy tp tn fp fn
        f1Value = f1score (asValue precisionValue :: Double) (asValue recallValue :: Double)

    putStrLn $ "Precision: " ++ show (asValue precisionValue :: Float)
    putStrLn $ "Recall: " ++ show (asValue recallValue :: Float)
    putStrLn $ "Accuracy: " ++ show (asValue accuracyValue :: Float)
    putStrLn $ "F1 Score: " ++ show (f1Value)
    --     tp = sumAll (mul tg pred)
    --     tn = sumAll (mul (1 - tg) (1 - pred))
    --     fp = sumAll (mul (1 - tg) pred)
    --     fn = sumAll (mul tg (1 - pred))
    -- putStrLn "1"
    -- putStrLn $ "tp : " ++ show (asValue tp :: Float)
    -- putStrLn $ "fp : " ++ show (asValue fp :: Float)
    -- let precccc = precision tp fp
    -- putStrLn "2"
    -- putStrLn $ "precision : " ++ show (asValue precccc :: Float)


main :: IO()
main = do
    trainData <- loadData "data/train.csv"
    evalData <- loadData "data/eval.csv"
    validData <- loadData "data/valid.csv"
    
    case (trainData, evalData, validData) of
        (Right trainDataset, Right evalDataset, Right validDataset) -> do
            let preModel = MLPspec
                    { 
                        feature_counts = [7, 32, 16, 8, 1],
                        nonlinearitySpec = Torch.sigmoid
                    }
            initModel <- sample preModel

            let (featuresTrain, targetsTrain) = (dataToTensor trainDataset, targetToTensor trainDataset)
            let (featuresEval, targetsEval) = (dataToTensor evalDataset, targetToTensor evalDataset)

            
            trainedModel <- train initModel featuresTrain targetsTrain 
            putStrLn "Training completed."

            evalModel <- train' trainedModel featuresEval targetsEval
            putStrLn "Eval Complet"
            eval evalModel featuresEval targetsEval

        (Left err, _, _) -> putStrLn ("Error in train file: " ++ show err)
        (_, Left err, _) -> putStrLn ("Error in evaluation file: " ++ show err)
        (_, _, Left err) -> putStrLn ("Error in validation file: " ++ show err)