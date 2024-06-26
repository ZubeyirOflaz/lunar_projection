file_dict = {
        256 : ["LDEM_256_00N_90N_000_180", 
                "LDEM_256_00N_90N_180_360", 
                "LDEM_256_00S_90S_000_180", 
                "LDEM_256_90S_00S_180_360"],
        
        512 : ["LDEM_512_00N_45N_000_090",
                "LDEM_512_00N_45N_090_180",
                "LDEM_512_00N_45N_180_270",
                "LDEM_512_00N_45N_270_360",
                "LDEM_512_45N_90N_000_090",
                "LDEM_512_45N_90N_090_180",
                "LDEM_512_45N_90N_180_270",
                "LDEM_512_45N_90N_270_360",
                "LDEM_512_45S_00S_000_090",
                "LDEM_512_45S_00S_090_180",
                "LDEM_512_45S_00S_180_270",
                "LDEM_512_45S_00S_270_360",
                "LDEM_512_90S_45S_000_090",
                "LDEM_512_90S_45S_090_180",
                "LDEM_512_90S_45S_180_270",
                "LDEM_512_90S_45S_270_360"]
    }

def retry_download(func):
    def wrapper(*args, **kwargs):
        num_retries = 3
        num_tried = 0
        while num_tried < num_retries:
            try:
                func(*args, **kwargs)
                break
            except Exception as e:
                print(f"An error occurred: {e}. Retrying...")
                num_tried += 1
                if num_tried == num_retries:
                    print(f"Failed to download {args[0]}.")
                    raise e
    return wrapper