module LinearRegression (linear, cost, calculateNewA, calculateNewB, train) where

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

train :: 
    Int ->    -- ^ remaining epochs
    Float ->  -- ^ learning rate
    Tensor -> -- ^ input features
    Tensor -> -- ^ target values
    (Tensor, Tensor) -> -- ^ current parameters (slope, intercept)
    IO (Tensor, Tensor) -- ^ final parameters (slope, intercept)
train 0 _ _ _ (slope, intercept) = return (slope, intercept)
train epochsLeft learningRate features targets (currentSlope, currentIntercept) =
  let newSlope = calculateNewA features targets currentSlope currentIntercept learningRate
      newIntercept = calculateNewB features targets currentSlope currentIntercept learningRate
      currentLoss = cost targets (linear (newSlope, newIntercept) features)
  in 
      train (epochsLeft-1) learningRate features targets (newSlope, newIntercept)

main :: IO ()
main = do
  let initialSlope = asTensor ([10] :: [Double])
  let initialIntercept = asTensor ([100] :: [Double])
  
  let initialPredictions = linear (initialSlope, initialIntercept) featureValues

  let predictionsList = asValue initialPredictions :: [Double]
  let actualValuesList = asValue targetValues :: [Double]

  let testNewSlope = asValue (calculateNewA featureValues targetValues initialSlope initialIntercept 0.1) :: Double
  let testNewIntercept = asValue (calculateNewB featureValues targetValues initialSlope initialIntercept 0.1) :: Double

  let totalEpochs = 100
  let optimalLearningRate = 2e-5

  mapM_ (\(actual, predicted) -> do
      putStrLn $ "Actual value: " ++ show actual
      putStrLn $ "Predicted value: " ++ show predicted
      putStrLn "**********")
      (zip actualValuesList predictionsList)

  let initialLoss = asValue (cost targetValues initialPredictions) :: Double
  putStrLn ("Initial loss: " ++ show initialLoss)
  putStrLn "**********"

  (finalSlope, finalIntercept) <- train totalEpochs optimalLearningRate featureValues targetValues (initialSlope, initialIntercept)
  
  putStrLn ("Target slope = 0.555, Final slope = " ++ show (asValue finalSlope :: Double))
  putStrLn ("Target intercept = 94.6, Final intercept = " ++ show (asValue finalIntercept :: Double))
  putStrLn "**********"

  return ()