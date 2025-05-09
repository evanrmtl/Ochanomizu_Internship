import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.TensorFactories (randIO')
import Torch.Functional.Internal (mul)
import Torch.Functional (sumAll)
import Control.Monad (mapM_)

trainingData :: [([Int],Int)]
trainingData = [([1,1],1),([1,0],0),([0,1],0),([0,0],0)]

step :: Tensor -> Tensor
step t = 
    if (asValue t :: Float) >= 0
        then asTensor (1 :: Float)
        else asTensor (0 :: Float)

perceptron ::
    Tensor -> -- x
    Tensor -> -- weights
    Tensor -> -- bias
    Tensor    -- output
perceptron x w b =
    let sumXW = sumAll (x `mul` w)
        tot = sumXW + b
    in step tot

calculateError :: Tensor -> Tensor -> Tensor
calculateError expected predicted = expected - predicted

calculateTotalError ::
    Tensor -> -- poids
    Tensor -> -- biais
    [([Int], Int)] ->
    Float
calculateTotalError weights bias dataset =
    let errors = map (\(inputs, label) ->
            let inputTensor = asTensor (map (\x -> fromIntegral x :: Float) inputs)
                expectedTensor = asTensor (fromIntegral label :: Float)
                output = perceptron inputTensor weights bias
                err = calculateError expectedTensor output
            in abs (asValue err :: Float)
            ) dataset
    in sum errors

updateWeights ::
    Tensor -> -- ^ poids
    Tensor -> -- ^ biais
    Tensor -> -- ^ entrée
    Tensor -> -- ^ étiquette attendue
    Float ->  -- ^ taux d'apprentissage
    (Tensor, Tensor) -- ^ nouveaux poids et biais
updateWeights weights bias input expected lr =
    let output = perceptron input weights bias
        error = expected - output
        lrTensor = asTensor lr
        newWeights = weights + mul (input * error) lrTensor
        newBias = bias + mul error lrTensor
    in (newWeights, newBias)

trainPerceptron ::
  Int -> -- ^ epochs
  Float -> -- ^ learning rate
  [([Int], Int)] -> -- ^ données d'entraînement
  Tensor -> -- ^ poids initiaux
  Tensor -> -- ^ biais initial
  IO (Tensor, Tensor) -- ^ poids et biais appris
trainPerceptron 0 _ _ weights bias = return (weights, bias)
trainPerceptron epochs lr dataset weights bias = do
    let totalError = calculateTotalError weights bias dataset
    putStrLn $ "Epoch " ++ show (50 - epochs + 1) ++ "/50"
    putStrLn $ "  Error: " ++ show totalError
    putStrLn $ "  Weights: " ++ show (asValue weights :: [Float])
    putStrLn $ "  Bias: " ++ show (asValue bias :: Float)
    
    let updated = foldl
            (\(w, b) (inputs, label) ->
            let inputTensor = asTensor (map (\x -> fromIntegral x :: Float) inputs)
                expectedTensor = asTensor (fromIntegral label :: Float)
            in updateWeights w b inputTensor expectedTensor lr)
            (weights, bias)
            dataset
    
    trainPerceptron (epochs - 1) lr dataset (fst updated) (snd updated)

main :: IO ()
main = do
    -- Initialisation aléatoire
    w1 <- randIO' [1] :: IO Tensor
    w2 <- randIO' [1] :: IO Tensor
    b <- randIO' [1] :: IO Tensor

    let weights = asTensor [asValue w1 :: Float, asValue w2 :: Float]
    let bias = asTensor (asValue b :: Float)

    let learningRate = 0.1 :: Float
    let epochs = 50

    putStrLn "Entraînement du perceptron pour la porte logique AND..."
    putStrLn "----------------------------------------------------"
    
    -- Entraînement
    (finalWeights, finalBias) <- trainPerceptron epochs learningRate trainingData weights bias

    putStrLn "----------------------------------------------------"
    putStrLn "Entraînement terminé !"
    putStrLn $ "Poids finaux: " ++ show (asValue finalWeights :: [Float])
    putStrLn $ "Biais final: " ++ show (asValue finalBias :: Float)

    -- Test
    putStrLn "\nRésultats sur les données d'entraînement:"
    putStrLn "----------------------------------------------------"
    mapM_ (\(inputs, label) -> do
        let inputTensor = asTensor (map (\x -> fromIntegral x :: Float) inputs)
            prediction = perceptron inputTensor finalWeights finalBias
            predictedVal = asValue prediction :: Float
        putStrLn $ "Entrée: " ++ show inputs ++ 
                   ", Attendu: " ++ show label ++ 
                   ", Prédit: " ++ show (round predictedVal)
        ) trainingData