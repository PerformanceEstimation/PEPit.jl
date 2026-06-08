@doc raw"""
    merge_dicts(dict1::OrderedDict, dict2::OrderedDict)

Merge two coefficient dictionaries by adding coefficients with matching keys.

This helper is used when symbolic points, functions, and expressions are added
together. The keys encode symbolic atoms and the values encode their scalar
coefficients.

# Arguments
- `dict1`: first coefficient dictionary.
- `dict2`: second coefficient dictionary.

# Returns
An `OrderedDict` containing the union of both key sets, with coefficients added
for keys that appear in both inputs.
"""
function merge_dicts(dict1::OrderedDict, dict2::OrderedDict)
    merged_dict = copy(dict1)
    for (key, value) in dict2
        merged_dict[key] = get(merged_dict, key, 0) + value
    end
    return merged_dict
end


@doc raw"""
    prune_dict(d::OrderedDict)

Remove all zero coefficients from a symbolic decomposition dictionary.

The symbolic algebra stores many affine and linear combinations as dictionaries.
After additions or scalar multiplications, exact zero coefficients can appear
and should be removed so that equality checks on decompositions remain stable.

# Arguments
- `d`: coefficient dictionary to prune.

# Returns
A new `OrderedDict` with all entries whose value is exactly zero removed.
"""
prune_dict(d::OrderedDict{K,V}) where {K,V} = OrderedDict{K,V}(k => v for (k, v) in d if v != 0)


@doc raw"""
    multiply_dicts(dict1::OrderedDict, dict2::OrderedDict)

Develop the product of two symbolic linear combinations.

If `dict1` represents `\sum_i a_i p_i` and `dict2` represents
`\sum_j b_j q_j`, then the returned dictionary represents the bilinear
expansion with keys `(p_i, q_j)` and coefficients `a_i b_j`. This is the
dictionary-level operation behind inner products of symbolic [`Point`](@ref)
objects.

# Arguments
- `dict1`: first coefficient dictionary.
- `dict2`: second coefficient dictionary.

# Returns
An `OrderedDict{Any,Float64}` whose keys are ordered pairs of input keys and
whose values are products of the corresponding coefficients, added when the same
pair appears more than once.
"""
function multiply_dicts(dict1::OrderedDict, dict2::OrderedDict)
    product_dict = OrderedDict{Any,Float64}()
    for (key1, value1) in dict1
        for (key2, value2) in dict2
            product_key = (key1, key2)
            product_value = value1 * value2
            product_dict[product_key] = get(product_dict, product_key, 0) + product_value
        end
    end
    return product_dict
end
