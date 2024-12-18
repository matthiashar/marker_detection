#ifndef CODES_H
#define CODES_H

#include <utility>
#include <vector>
#include <array>

namespace marker_detection {
	
	/**
	 * @brief Function to find the ID to a bit code.
	 * @param code 
	 * @param bit 
	 * @param use_metashape 
	 * @return 
	 */
	int findID(const uint32_t code, const int bit, const bool use_metashape = false);

    /**
     * @brief Perform a bitwise rotation to the left
     * @param val 
     * @param bits 
     * @param total_bits 
     * @return 
     */
    uint32_t rotateBitwiseLeft(uint32_t val, int bits, int total_bits);

    /**
     * @brief Check all bitwise rotations to find smallest representation
     * @param val 
     * @param total_bits 
     * @return 
     */
    uint32_t findSmallestRotation(uint32_t val, int total_bits);

    /**
     * @brief Calculate parity (returns true if even parity)
     * @param val 
     * @return 
     */
    bool calculateParity(uint32_t val);

    /**
     * @brief Count number of bit transitions
     * @param val 
     * @return 
     */
    int countBitTransitions(uint32_t val);

    /**
     * @brief Generate codes for a given number of bits and, optionally, a given number of transitions
     * @tparam Bits 
     * @return 
     */
    template <int Bits>
    std::array<uint32_t, (1 << (Bits - 2))> generateCodes();

	/**
	 * @brief Function to create a map for converting the default ID�s to Metashape ID�s (for using default markers in metashape).
	 * @param bit (only 12 or 14)
	 * @param map for conversion of ID�s (key=ID, value=MetashapeID)
	 */
	void createId2MetashapeConversionMap(int bit, std::vector<std::pair<int, int>>& map);

    /**
     * @brief Return id to code in array of codes (id = index + 1)
     * @tparam N 
     * @param code 
     * @param codes 
     * @return 
     */
    template <size_t N>
    int findID(uint32_t code, const std::array<uint32_t, N>& codes) {
        for (size_t i = 0; i < N; ++i) {
            if (codes[i] == code) {
                return static_cast<int>(i) + 1;
            }
            else if (codes[i] > code) {
                break; 
            }
        }
        return -1;
    }

	// Look up table for 12 bit codes (id = index + 1)
    const auto code2id12 = generateCodes<12>();

	// Look up table for 14 bit codes (id = index + 1)
    const auto code2id14 = generateCodes<14>();

	// Look up table for 14 bit Metashape codes (id = index + 1)
	constexpr std::array<uint32_t, 561> code2metashape14 = { 135, 139, 141, 147, 149, 153, 159, 163, 165, 169, 175, 177, 183, 187,
		189, 195, 197, 201, 207, 209, 215, 219, 221, 225, 231, 235, 237, 243, 245, 249, 255, 263, 267, 269, 275, 277,
		281, 287, 291, 293, 297, 303, 305, 311, 315, 317, 323, 325, 329, 335, 337, 343, 347, 349, 353, 359, 363, 365,
		371, 373, 377, 383, 387, 389, 393, 399, 401, 407, 411, 413, 417, 423, 427, 429, 435, 437, 441, 447, 455, 459,
		461, 467, 469, 473, 479, 483, 485, 489, 495, 497, 503, 507, 509, 531, 533, 537, 543, 547, 549, 553, 559, 561,
		567, 571, 573, 579, 581, 585, 591, 593, 599, 603, 605, 615, 619, 621, 627, 629, 633, 639, 645, 649, 655, 657,
		663, 667, 669, 679, 683, 685, 691, 693, 697, 703, 711, 715, 717, 723, 725, 729, 735, 739, 741, 745, 751, 753,
		759, 763, 765, 783, 785, 791, 795, 797, 807, 811, 813, 819, 821, 825, 831, 839, 843, 845, 851, 853, 857, 863,
		867, 869, 873, 879, 881, 887, 891, 893, 903, 907, 909, 915, 917, 921, 927, 931, 933, 937, 943, 945, 951, 955,
		957, 965, 969, 975, 977, 983, 987, 989, 999, 1003, 1005, 1011, 1013, 1017, 1023, 1093, 1097, 1103, 1111, 1115,
		1117, 1127, 1131, 1133, 1139, 1141, 1145, 1151, 1161, 1167, 1175, 1179, 1181, 1191, 1195, 1197, 1203, 1205,
		1209, 1215, 1223, 1227, 1229, 1235, 1237, 1241, 1247, 1251, 1253, 1257, 1263, 1271, 1275, 1277, 1303, 1307,
		1309, 1319, 1323, 1325, 1331, 1333, 1337, 1343, 1351, 1355, 1357, 1363, 1365, 1369, 1375, 1379, 1381, 1385,
		1391, 1399, 1403, 1405, 1419, 1421, 1427, 1429, 1433, 1439, 1443, 1445, 1449, 1455, 1463, 1467, 1469, 1481,
		1487, 1495, 1499, 1501, 1511, 1515, 1517, 1523, 1525, 1529, 1535, 1587, 1589, 1593, 1599, 1607, 1611, 1613,
		1619, 1621, 1625, 1631, 1637, 1641, 1647, 1655, 1659, 1661, 1677, 1683, 1685, 1689, 1695, 1701, 1705, 1711,
		1719, 1723, 1725, 1737, 1743, 1751, 1755, 1757, 1767, 1771, 1773, 1779, 1781, 1785, 1791, 1823, 1829, 1833,
		1839, 1847, 1851, 1853, 1865, 1871, 1879, 1883, 1885, 1895, 1899, 1901, 1907, 1909, 1913, 1919, 1935, 1943,
		1947, 1949, 1959, 1963, 1965, 1971, 1973, 1977, 1983, 1995, 1997, 2003, 2005, 2009, 2015, 2021, 2025, 2031,
		2039, 2043, 2045, 2343, 2347, 2349, 2355, 2357, 2367, 2379, 2381, 2387, 2389, 2399, 2405, 2415, 2423, 2427,
		2429, 2451, 2453, 2463, 2469, 2479, 2487, 2491, 2493, 2511, 2519, 2523, 2525, 2535, 2539, 2541, 2547, 2549,
		2559, 2643, 2645, 2655, 2671, 2679, 2683, 2685, 2709, 2719, 2735, 2743, 2747, 2749, 2767, 2775, 2779, 2781,
		2791, 2795, 2797, 2803, 2805, 2815, 2863, 2871, 2875, 2877, 2895, 2903, 2907, 2909, 2919, 2923, 2925, 2931,
		2933, 2943, 2967, 2971, 2973, 2983, 2987, 2989, 2995, 2997, 3007, 3021, 3027, 3029, 3039, 3055, 3063, 3067,
		3069, 3279, 3287, 3291, 3293, 3303, 3307, 3309, 3317, 3327, 3383, 3387, 3389, 3407, 3415, 3419, 3421, 3431,
		3435, 3437, 3445, 3455, 3483, 3485, 3495, 3499, 3501, 3509, 3519, 3541, 3551, 3567, 3575, 3579, 3581, 3701,
		3711, 3741, 3755, 3757, 3765, 3775, 3797, 3807, 3823, 3831, 3835, 3837, 3903, 3925, 3935, 3951, 3959, 3963,
		3965, 3999, 4015, 4023, 4027, 4029, 4055, 4059, 4061, 4075, 4077, 4085, 4095, 5463, 5467, 5483, 5503, 5547,
		5567, 5599, 5615, 5623, 5627, 5823, 5855, 5871, 5879, 5883, 5983, 5999, 6007, 6011, 6063, 6071, 6075, 6107,
		6143, 7023, 7031, 7095, 7167, 7679, 7935, 8063, 8127 };

	// Look up table for 12 bit Metashape codes (id = index + 1)
	constexpr std::array<uint32_t, 161> code2metashape12 = { 71, 75, 77, 83, 85, 89, 95, 99, 101, 105, 111, 113, 119, 123, 125, 135,
		139, 141, 147, 149, 153, 159, 163, 165, 169, 175, 177, 183, 187, 189, 195, 197, 201, 207, 209, 215, 219, 221,
		231, 235, 237, 243, 245, 249, 255, 275, 277, 281, 287, 291, 293, 297, 303, 311, 315, 317, 325, 329, 335, 343,
		347, 349, 359, 363, 365, 371, 373, 377, 383, 399, 407, 411, 413, 423, 427, 429, 435, 437, 441, 447, 455, 459,
		461, 467, 469, 473, 479, 485, 489, 495, 503, 507, 509, 585, 591, 599, 603, 605, 615, 619, 621, 627, 629, 639,
		663, 667, 669, 679, 683, 685, 691, 693, 703, 715, 717, 723, 725, 735, 751, 759, 763, 765, 819, 821, 831, 845,
		853, 863, 879, 887, 891, 893, 927, 943, 951, 955, 957, 975, 983, 987, 989, 1003, 1005, 1013, 1023, 1365, 1375,
		1391, 1399, 1403, 1455, 1463, 1467, 1495, 1499, 1535, 1755, 1791, 1919, 1983, 2015 };
}

#endif