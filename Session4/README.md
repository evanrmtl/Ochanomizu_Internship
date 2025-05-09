# AND

## code

```haskell
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
```

## Result

Epoch 1/50
  Error: 3.0
  Weights: [0.7746462,0.95062184]
  Bias: 0.65297997
Epoch 2/50
  Error: 3.0
  Weights: [0.6746462,0.8506218]
  Bias: 0.35297996
Epoch 3/50
  Error: 3.0
  Weights: [0.5746462,0.7506218]
  Bias: 5.297997e-2
Epoch 4/50
  Error: 2.0
  Weights: [0.47464618,0.6506218]
  Bias: -0.14702004
Epoch 5/50
  Error: 2.0
  Weights: [0.3746462,0.55062175]
  Bias: -0.34702003
Epoch 6/50
  Error: 0.0
  Weights: [0.2746462,0.45062175]
  Bias: -0.54702

...


<p>Epoch 50/50<br>
  Error: 0.0<br>
  Weights: [0.2746462,0.45062175]<br>
  Bias: -0.54702<br></p>


Entraînement terminé !
Poids finaux: [0.2746462,0.45062175]
Biais final: -0.54702

Résultats sur les données d'entraînement:
----------------------------------------------------
Entrée: [1,1], Attendu: 1, Prédit: 1
Entrée: [1,0], Attendu: 0, Prédit: 0
Entrée: [0,1], Attendu: 0, Prédit: 0
Entrée: [0,0], Attendu: 0, Prédit: 0


# XOR

A forward function computes predictions: y = x * w + b 
(perceptron function)
The loss function is Mean Squared Error between prediction and target.
Backpropagation is handled with the function runStep, who return the new state.
