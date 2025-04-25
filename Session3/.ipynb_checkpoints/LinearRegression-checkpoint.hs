import Torch.Tensor (Tensor, asTensor)
import Torch.Functional (matmul, add, transpose2D)


ys :: Tensor
ys = [130, 195, 218, 166, 163, 155, 204, 270, 205, 127, 260, 249, 251, 158, 167]
xs :: Tensor
xs = [148, 186, 279, 179, 216, 127, 152, 196, 126, 78, 211, 259, 255, 115, 173]

linear :: 
    (Tensor, Tensor) -> -- ^ parameters ([a, b]: 1 × 2, c: scalar)
    Tensor ->           -- ^ data x: 1 × 10
    Tensor              -- ^ z: 1 × 10
linear (slope, intercept) input = add intercept $ matmul input slope


main :: IO ()
main = do
  let sampleA = convertToTensor 0.555
  let sampleB = convertToTensor 94.585026

  -- linear (sampleA sampleB) . foldl xs
  map (linear (sampleA sampleB) ) xs
  map

  
  -- Below are pseudo code
  -- sampleA = convertToTensor 0.555
	-- sampleB = convertToTensor 94.585026
	
  -- Iterate through the provided xs and ys data.
  -- For each pair, convert x to a tensor, calculate the estimatedY using your linear function with the provided sampleA and sampleB, 
  -- and print both the correct y and the estimatedY.
  
  --for x, y in (xs, xy)
  --  convertToTensor x
	--  estimatedY = linear (sampleA, sampleB) x
	--  print "correct answer:" + y
	--  print "estimated: " + estimatedY
	--  print "******"
	 
	-- Expected outputs:
	-- correct answer: 148
	-- estimated: ?
	-- *******
	-- correct answer: 186
	-- ...
  --return ()