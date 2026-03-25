import array
import math
import struct
from typing import List

class OptPForDeltaPostings:
    """
    OptPForDelta-style postings codec that follows the classroom idea:

    At first, split numbers into fixed-size blocks and choose the smallest bit width b such that the number of exceptions 
    is at most about 10% of the block (based on Lecture's slides). Then, we store non-exception values in a packed fixed-width bitstream.
    We also have to store exception positions and exception values separately.

    DocIDs are first converted into d-gaps for posting lists and raw TF values are compressed directly for TF List.
    """

    BLOCK_SIZE = 128
    EXCEPTION_RATIO = 0.10

    @staticmethod
    def encode(postings_list: List[int]) -> bytes:
        if not postings_list:
            return b""

        gaps = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gaps.append(postings_list[i] - postings_list[i - 1])

        return OptPForDeltaPostings._encode_numbers(gaps)

    @staticmethod
    def decode(encoded_postings_list: bytes) -> List[int]:
        if not encoded_postings_list:
            return []

        gaps = OptPForDeltaPostings._decode_numbers(encoded_postings_list)
        if not gaps:
            return []

        postings = [gaps[0]]
        for i in range(1, len(gaps)):
            postings.append(postings[-1] + gaps[i])

        return postings

    @staticmethod
    def encode_tf(tf_list: List[int]) -> bytes:
        return OptPForDeltaPostings._encode_numbers(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list: bytes) -> List[int]:
        return OptPForDeltaPostings._decode_numbers(encoded_tf_list)

    @staticmethod
    def _bits_required(x: int) -> int:
        return max(1, x.bit_length())

    @staticmethod
    def _choose_b(block: List[int]) -> int:
        """
        Choose the smallest bit width b such that the number of exceptions
        is <= ceil(10% of block size), while avoiding the degenerate case
        where everything becomes an exception.
        """
        n = len(block)
        if n == 0:
            return 1

        max_bits = max(OptPForDeltaPostings._bits_required(x) for x in block)
        allowed_exceptions = max(1, math.ceil(OptPForDeltaPostings.EXCEPTION_RATIO * n))

        for b in range(1, max_bits + 1):
            threshold = 1 << b
            exc_count = sum(1 for x in block if x >= threshold)

            if exc_count <= allowed_exceptions and exc_count < n:
                return b

        return max_bits

    @staticmethod
    def _pack_fixed_width(values: List[int], b: int) -> bytes:
        """
        Pack a list of non-negative integers using exactly b bits each.
        """
        if not values:
            return b""

        out = bytearray()
        bit_buffer = 0
        bit_count = 0
        mask = (1 << b) - 1

        for value in values:
            if value < 0:
                raise ValueError("Only non-negative integers are supported.")

            bit_buffer = (bit_buffer << b) | (value & mask)
            bit_count += b

            while bit_count >= 8:
                shift = bit_count - 8
                out.append((bit_buffer >> shift) & 0xFF)
                bit_buffer &= (1 << shift) - 1 if shift > 0 else 0
                bit_count -= 8

        if bit_count > 0:
            out.append((bit_buffer << (8 - bit_count)) & 0xFF)

        return bytes(out)

    @staticmethod
    def _unpack_fixed_width(data: bytes, count: int, b: int) -> List[int]:
        """
        Unpack `count` integers from a bytestream where each integer uses b bits.
        """
        if count == 0:
            return []

        total_bits = len(data) * 8
        needed_bits = count * b
        if total_bits < needed_bits:
            raise ValueError("Packed data is shorter than expected.")

        acc = int.from_bytes(data, byteorder="big")

        pad_bits = total_bits - needed_bits
        if pad_bits > 0:
            acc >>= pad_bits

        mask = (1 << b) - 1
        values = []
        for i in range(count):
            shift = (count - 1 - i) * b
            values.append((acc >> shift) & mask)

        return values

    @staticmethod
    def _encode_numbers(numbers: List[int]) -> bytes:
        """
        Encode numbers block by block.

        Global format:
        - uint32 total_count

        Per block format:
        - uint16 n               : number of integers in this block
        - uint8  b               : chosen bit width
        - uint8  e               : number of exceptions
        - e x uint8              : exception positions
        - packed normal values   : ceil((n - e) * b / 8) bytes
        - e x uint32             : exception values

        Notes:
        - Because BLOCK_SIZE = 128, positions fit in uint8.
        - Content stores ONLY non-exception values, matching the slide idea.
        """
        if not numbers:
            return b""

        out = bytearray()
        out.extend(struct.pack(">I", len(numbers)))

        for start in range(0, len(numbers), OptPForDeltaPostings.BLOCK_SIZE):
            block = numbers[start:start + OptPForDeltaPostings.BLOCK_SIZE]
            n = len(block)
            b = OptPForDeltaPostings._choose_b(block)
            threshold = 1 << b

            exception_positions = []
            exception_values = []
            normal_values = []

            for idx, value in enumerate(block):
                if value < threshold:
                    normal_values.append(value)
                else:
                    exception_positions.append(idx)
                    exception_values.append(value)

            e = len(exception_positions)

            out.extend(struct.pack(">HBB", n, b, e))

            if e > 0:
                out.extend(bytes(exception_positions))

            packed_normals = OptPForDeltaPostings._pack_fixed_width(normal_values, b)
            out.extend(packed_normals)

            for value in exception_values:
                out.extend(struct.pack(">I", value))

        return bytes(out)

    @staticmethod
    def _decode_numbers(data: bytes) -> List[int]:
        """
        Decode the bytestream generated by `_encode_numbers`.
        """
        if not data:
            return []

        offset = 0
        total_count = struct.unpack_from(">I", data, offset)[0]
        offset += 4

        result = []

        while len(result) < total_count:
            n, b, e = struct.unpack_from(">HBB", data, offset)
            offset += struct.calcsize(">HBB")

            exception_positions = list(data[offset:offset + e])
            offset += e

            normal_count = n - e
            packed_len = (normal_count * b + 7) // 8
            packed_normals = data[offset:offset + packed_len]
            offset += packed_len

            normal_values = OptPForDeltaPostings._unpack_fixed_width(
                packed_normals, normal_count, b
            )

            exception_values = []
            for _ in range(e):
                value = struct.unpack_from(">I", data, offset)[0]
                offset += 4
                exception_values.append(value)

            exc_map = {pos: val for pos, val in zip(exception_positions, exception_values)}

            block = []
            normal_idx = 0
            for i in range(n):
                if i in exc_map:
                    block.append(exc_map[i])
                else:
                    block.append(normal_values[normal_idx])
                    normal_idx += 1

            result.extend(block)

        return result[:total_count]

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend ke depan
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # bit awal pada byte terakhir diganti 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)

if __name__ == '__main__':
    
    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings, OptPForDeltaPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("byte hasil encode postings: ", encoded_postings_list)
        print("ukuran encoded postings   : ", len(encoded_postings_list), "bytes")
        print("byte hasil encode TF list : ", encoded_tf_list)
        print("ukuran encoded TF list   : ", len(encoded_tf_list), "bytes")
        
        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("hasil decoding (postings): ", decoded_posting_list)
        print("hasil decoding (TF list) : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, "hasil decoding tidak sama dengan postings original"
        assert decoded_tf_list == tf_list, "hasil decoding tidak sama dengan postings original"
        print()
