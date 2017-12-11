""" quiz materials for feature scaling clustering """

### FYI, the most straightforward implementation might
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!
### why would you rescale it?  Or even use it at all?
def featureScaling(arr):
    m = max(arr) - min(arr)
    new = []
    if m != 0:
        for i in arr:
            new.append((i - min(arr)) / float(m))

    return new

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)
salary = [477,1111258,200000]
exercised_stock_options = [34348384,3285,1000000]

print featureScaling(salary)
print featureScaling(exercised_stock_options)