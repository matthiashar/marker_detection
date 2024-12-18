#include "codes.h"

#include <iostream>

namespace marker_detection {

	int findID(const uint32_t code, const int bit, const bool use_metashape)
	{
		if (use_metashape) {
			if (bit == 12)
				return findID(code, code2metashape12);
			if (bit == 14)
				return findID(code, code2metashape14);
		}
		else
		{
			if (bit == 12)
				return findID(code, code2id12);
			if (bit == 14)
				return findID(code, code2id14);
		}
		return -1;
	}


    uint32_t rotateBitwiseLeft(uint32_t val, int bits, int total_bits) {
        return ((val << bits) & ((1u << total_bits) - 1)) | (val >> (total_bits - bits));
    }


    uint32_t findSmallestRotation(uint32_t val, int total_bits) {
        uint32_t smallest = val;
        for (int i = 1; i < total_bits; ++i) {
            smallest = std::min(rotateBitwiseLeft(val, i, total_bits), smallest);
        }
        return smallest;
    }


    bool calculateParity(uint32_t val) {
        bool parity = true;
        while (val) {
            parity = !parity;
            val &= (val - 1);
        }
        return parity;
    }


    int countBitTransitions(uint32_t val) {
        int transitions = 0;
        int prev_bit = val & 1;
        while (val >>= 1) {
            int new_bit = val & 1;
            if (new_bit != prev_bit) {
                ++transitions;
            }
            prev_bit = new_bit;
        }
        return transitions;
    }

    template <int Bits>
    std::array<uint32_t, (1 << (Bits - 2))> generateCodes() {
        int transitions = -1;
        std::array<uint32_t, (1 << (Bits - 2))> codes = {};
        int index = 0;
        int half_bits = Bits / 2;

        for (uint32_t i = 0; i < (1u << (Bits - 2)); ++i) {
            uint32_t code = (i << 1) | 1;
            code = findSmallestRotation(code, Bits);

            uint32_t diff = (code & ((1 << half_bits) - 1)) &
                ((code >> half_bits) & ((1 << half_bits) - 1));

            bool parity = calculateParity(code);
            int num_transitions = transitions >= 0 ? countBitTransitions(code) : -1;

            if (parity && diff > 0 && (transitions < 0 || transitions == num_transitions)) {
                if (std::find(codes.begin(), codes.end(), code) == codes.end()) {
                    codes[index++] = code;
                }
            }
        }
        return codes;
    }

    void createId2MetashapeConversionMap(int bit, std::vector<std::pair<int, int>>& map)
    {
        for (int i = 0; i < 8200; i++) {
            int id = marker_detection::findID(i, bit, false);
            int id_meta = marker_detection::findID(i, bit, true);
            if (id > 0) {
                map.push_back(std::make_pair(id, id_meta));
                std::cout << i << " " << id << " " << id_meta << std::endl;
            }
        }
    }
}