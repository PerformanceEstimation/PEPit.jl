function merge_dicts(dict1::OrderedDict, dict2::OrderedDict)
    merged_dict = copy(dict1)
    for (key, value) in dict2
        merged_dict[key] = get(merged_dict, key, 0) + value
    end
    return merged_dict
end


prune_dict(d::OrderedDict{K,V}) where {K,V} = OrderedDict{K,V}(k => v for (k, v) in d if v != 0)


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
