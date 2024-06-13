if __name__ == '__main__':
    def copy_first_1gb(source_path, destination_path):
        chunk_size = 1024 * 1024  # 1 MB
        total_size = 1024 * 1024 * 1024  # 1 GB
        copied_size = 0

        with open(source_path, 'rb') as src_file:
            with open(destination_path, 'wb') as dst_file:
                while copied_size < total_size:
                    if total_size - copied_size < chunk_size:
                        chunk_size = total_size - copied_size

                    data = src_file.read(chunk_size)
                    if not data:
                        break

                    dst_file.write(data)
                    copied_size += len(data)
                    print(f"Copied {copied_size} bytes", end='\r')

        print(f"Finished copying first 1 GB from {source_path} to {destination_path}")


    # Example usage
    source_file = 'data/raw/lichess_db_standard_rated_2019-01.pgn'
    destination_file = 'data/raw/lichess_db_standard_rated_2019-01-smaller.pgn'
    copy_first_1gb(source_file, destination_file)
