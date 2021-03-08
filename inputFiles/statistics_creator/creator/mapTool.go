package creator

import (
	"fmt"
	"strconv"
)

func mapGet(dict map[string]interface{}, key string) interface{} {
	ret, ok := dict[key]
	if !ok {
		panic(fmt.Errorf("Key (%s) did not found in the file", key))
	}
	return ret
}

func mapGetString(dict map[string]interface{}, key string) string {
	tmp, ok := dict[key]
	if !ok {
		panic(fmt.Errorf("Key (%s) did not found in the file", key))
	}
	switch v := tmp.(type) {
	case string:
		return v
	case float64:
		return strconv.FormatFloat(v, 'f', 0, 64)
	default:
		panic(fmt.Errorf("Key (%s) could not transform directly or indirectly to string", key))
	}
}

func mapGetInt(dict map[string]interface{}, key string) int {
	tmp, ok := dict[key]
	if !ok {
		panic(fmt.Errorf("Key (%s) did not found in the file", key))
	}
	switch v := tmp.(type) {
	case float64:
		return int(v)
	case string:
		vInteger, err := strconv.Atoi(v)
		if err != nil {
			panic(fmt.Errorf("String to int conversion error for key (%s), error is : %v", key, err))
		}
		return vInteger
	default:
		panic(fmt.Errorf("Key (%s) could not transform directly or indirectly to int", key))
	}
}
