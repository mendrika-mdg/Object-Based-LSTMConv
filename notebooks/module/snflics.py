import regex as re              
import os
import numpy as np
from netCDF4 import Dataset     
import netCDF4 as nc            
from scipy import ndimage       
from skimage import measure    


def prepare_core(file, spatial_filter_size, ymin, ymax, xmin, xmax):
    """
    Prepares core data from a NetCDF file by applying binary thresholding and maximum 
    filtering. Replaces NaN values with 0, sets negative values to 0, and converts 
    positive values to 1 before applying the spatial filter.

    Args:
        file (str): Path to the NetCDF file containing core data.
        spatial_filter_size (int): Size of the spatial filter to apply for smoothing.
        ymin (int): Minimum y index for slicing the core data.
        ymax (int): Maximum y index for slicing the core data.
        xmin (int): Minimum x index for slicing the core data.
        xmax (int): Maximum x index for slicing the core data.

    Returns:
        np.ndarray: Binary array of processed core data after thresholding and filtering.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the spatial filter size is not a positive integer.
        RuntimeError: If the input data does not have the expected shape.
        IndexError: If the specified indices are out of bounds for the core data.
        OSError: If there is an issue opening the NetCDF file.
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"The file '{file}' does not exist.")
    
    if not isinstance(spatial_filter_size, int) or spatial_filter_size <= 0:
        raise ValueError("The spatial filter size must be a positive integer.")
    
    try:
        # Open the NetCDF file using a context manager to ensure proper file closure
        with Dataset(file, "r") as data:
            cores = data.variables["cores"][0, :, :]  # Accessing the variable safely
    except OSError as e:
        raise OSError(f"Error opening NetCDF file: {file}. {e}")
    
    # Ensure cores is a 2D array
    if cores.ndim != 2:
        raise RuntimeError("Input data must be a 2D array.")

    # Validate the slicing indices
    if not (0 <= ymin < ymax < cores.shape[0] and 0 <= xmin < xmax < cores.shape[1]):
        raise IndexError("Slicing indices are out of bounds for the core data.")

    # Slice the core data to the specified region
    cores = cores[ymin:ymax+1, xmin:xmax+1]

    # Replace NaN and negative values, then binarize the data
    cores = np.nan_to_num(cores, nan=0.0)  # Replace NaNs with 0
    cores = np.clip(cores, 0, 1)           # Set negative values to 0, positives to 1

    # Apply maximum filter for smoothing
    cores = ndimage.maximum_filter(cores, size=(spatial_filter_size, spatial_filter_size))
    
    return cores



def all_files_in(data_path):
    """
    Collects the paths of all files in the specified directory and its subdirectories.

    Args:
        data_path (str): The path of the directory to search for files.

    Returns:
        list: A list containing the full paths of all files found within the directory.
        
    Raises:
        ValueError: If the specified path is not a directory.
    """
    if not os.path.isdir(data_path):
        raise ValueError(f"The specified path '{data_path}' is not a valid directory.")
    
    all_files = []
    for dir_path, _, file_names in os.walk(data_path):
        for file_name in file_names:
            all_files.append(os.path.join(dir_path, file_name))
    
    return all_files


def get_time(filename):
    """
    Extracts the date and time from the filename of an observation file.
    The filename can contain any characters, including special characters or slashes,
    before the date format.

    Args:
        filename (str): The name of the file (must include date in the format YYYYMMDDHHMM.nc).

    Returns:
        dict: A dictionary containing the extracted year, month, day, hour, and minute.

    Raises:
        ValueError: If the filename does not match the expected date format.
    """
    # Use regular expression to match the expected date format in the filename
    match = re.search(r"[^/\\]*?(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})\.nc$", filename)

    if not match:
        raise ValueError(f"The filename '{filename}' does not contain a valid date format.")

    # Extract components from the regex match groups
    year, month, day, hour, minute = match.groups()

    return {
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute
    }


def get_date_as_path(filename):
    """
    Extracts a date-based path segment from the given filename.

    The filename should match the pattern '...YYYYMMDD.nc', where '...'
    can contain alphanumeric characters, hyphens, and slashes.

    Args:
        filename (str): The name of the file (must match the expected pattern).

    Returns:
        str: A modified path segment based on the extracted date.

    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    # Use regular expression to match the expected filename pattern
    match = re.search(r"([\w\d\-/]+)(\d+)\.nc$", filename)

    if not match:
        raise ValueError(f"The filename '{filename}' does not match the expected pattern.")

    # Extract components from the regex match groups
    day_path = match.group(1)
    date = match.group(2)

    # Construct and return the modified path segment
    return f"{day_path}{date[:-4]}"


def search(hour, minute, data_path):
    """
    Searches for files in the specified directory that match the given hour and minute.

    Args:
        hour (str): The hour to match in the filename (should be a two-digit string).
        minute (str): The minute to match in the filename (should be a two-digit string).
        data_path (str): The directory path to search for files.

    Returns:
        list: A list of file paths that match the specified hour and minute.

    Raises:
        ValueError: If hour or minute is not a valid two-digit string.
    """
    # Validate hour and minute
    if not (isinstance(hour, str) and hour.isdigit() and len(hour) == 2):
        raise ValueError(f"Invalid hour: {hour}. It should be a two-digit string.")
    if not (isinstance(minute, str) and minute.isdigit() and len(minute) == 2):
        raise ValueError(f"Invalid minute: {minute}. It should be a two-digit string.")

    all_files = all_files_in(data_path)
    filtering_result = []

    for file in all_files:
        # Extract time components from the filename
        file_time = get_time(file)
        if file_time["minute"] == minute and file_time["hour"] == hour:
            if file_time["month"] in ["11", "12", "01","02", "03", "04"]:
                filtering_result.append(file)

    return filtering_result


def search_ymd(year, month, day, data_path):
    """
    Searches for files in the specified directory that match the given year, month, and day.

    Args:
        year (str): The year to match in the filename (should be a four-digit string).
        month (str): The month to match in the filename (should be a two-digit string).
        day (str): The day to match in the filename (should be a two-digit string).
        data_path (str): The directory path to search for files.

    Returns:
        list: A list of file paths that match the specified year, month, and day.

    Raises:
        ValueError: If year is not a valid four-digit string, or if month or day are not valid two-digit strings.
    """
    # Validate year, month, and day
    if not (isinstance(year, str) and year.isdigit() and len(year) == 4):
        raise ValueError(f"Invalid year: {year}. It should be a four-digit string.")
    if not (isinstance(month, str) and month.isdigit() and len(month) == 2):
        raise ValueError(f"Invalid month: {month}. It should be a two-digit string.")
    if not (isinstance(day, str) and day.isdigit() and len(day) == 2):
        raise ValueError(f"Invalid day: {day}. It should be a two-digit string.")

    all_files = all_files_in(data_path)
    filtering_result = []

    for file in all_files:
        # Extract time components from the filename
        file_time = get_time(file)
        if (file_time["day"] == day and 
            file_time["month"] == month and 
            file_time["year"] == year):
            filtering_result.append(file)

    return filtering_result


def compute_pc(dataset, spatial_filter_size, ymin, ymax, xmin, xmax):
    """
    Computes the probabilistic core (PC) from the given dataset of files.

    Args:
        dataset (list): A list of file paths to the datasets.
        spatial_filter_size (int): The size of the spatial filter to apply.
        ymin (int): The minimum y index for slicing the core data.
        ymax (int): The maximum y index for slicing the core data.
        xmin (int): The minimum x index for slicing the core data.
        xmax (int): The maximum x index for slicing the core data.

    Returns:
        np.ndarray: The computed probabilistic core (PC) as a 2D array.
        
    Raises:
        ValueError: If no valid files were processed from the dataset.
    """
    # Initialize an array to accumulate core values
    sum_cores = np.zeros((ymax - ymin + 1, xmax - xmin + 1))  # Adjust shape based on slicing
    valid_files = 0  # Keep track of how many valid files were processed

    for file in dataset:
        try:
            # Prepare the cores using the prepare_core function with specified slicing
            cores = prepare_core(file, spatial_filter_size, ymin, ymax, xmin, xmax)
            # Accumulate the cores
            sum_cores += cores
            valid_files += 1  # Count this file as successfully processed
        except (FileNotFoundError, OSError, RuntimeError, IndexError) as e:
            # Log and skip corrupted or missing files
            print(f"Skipping {file} due to error: {e}")
            continue

    # Check if any valid files were processed
    if valid_files == 0:
        raise ValueError("No valid files were processed from the dataset.")

    # Compute the probabilistic core (PC)
    pc = sum_cores / valid_files  # Divide by the number of successfully processed files
    return pc



def core_index(core_value, msg_core):
    """
    Finds the indices of a specific core value in the given core array.

    Args:
        core_value (int or float): The value for which to find the indices.
        msg_core (np.ndarray): A 2D array where the core value will be searched.

    Returns:
        tuple: A tuple containing the y (row) and x (column) indices of the core value.

    Raises:
        ValueError: If the core value is not found in the array.
        IndexError: If the input array is empty.
    """
    if msg_core.size == 0:
        raise IndexError("The input array is empty.")
    
    # Find the indices of the specified core value
    index_core = np.argwhere(msg_core == core_value)

    if index_core.size == 0:
        raise ValueError(f"The core value '{core_value}' was not found in the array.")
    
    # Return the first occurrence of the core value
    y_core, x_core = index_core[0]
    return y_core, x_core


def to_yx(lat, lon, lats, lons):    
    """
    Finds the approximate grid indices (y, x) corresponding to a given latitude and longitude 
    in a grid defined by `lats` and `lons` arrays, with a tolerance of approximately 3 km 
    around the specified latitude and longitude.

    Parameters:
    ----------
    lat : float
        The target latitude (in degrees) for which to find the nearest grid point.
    lon : float
        The target longitude (in degrees) for which to find the nearest grid point.
    lats : numpy.ndarray
        A 2D array of latitudes representing the grid.
    lons : numpy.ndarray
        A 2D array of longitudes representing the grid.
    
    Returns:
    -------
    y : int
        The index along the latitude dimension of the grid where the target latitude is found.
    x : int
        The index along the longitude dimension of the grid where the target longitude is found.
    """
    
    # Calculate the absolute difference from the target latitude and longitude
    lat_diff = np.abs(lats - lat)
    lon_diff = np.abs(lons - lon)

    # Calculate the tolerance in degrees for 3 km (based on the approximations)
    lat_tol = 3 / 111  # 1 degree of latitude ~ 111 km
    lon_tol = 3 / 104  # 1 degree of longitude ~ 104 km (approximated)

    # Find the indices where the differences are within the tolerance
    lat_mask = lat_diff <= lat_tol
    lon_mask = lon_diff <= lon_tol

    # Combine the latitude and longitude masks to find the matching grid points
    valid_indices = np.argwhere(lat_mask & lon_mask)

    if valid_indices.size > 0:
        # If matching indices are found, return the median indices
        y, x = np.median(valid_indices, axis=0).astype(int)
        return y, x
    else:
        # If no matching points are found, raise an error or handle gracefully
        raise ValueError("No matching grid point found within the tolerance")



def identify_H0(hour, minute, data_path, Sy_min, Sy_max, Sx_min, Sx_max):
    """
    Identifies files containing cores within specified spatial boundaries 
    and time conditions.

    Args:
        hour (int): The hour to filter files.
        minute (int): The minute to filter files.
        data_path (str): The path to the directory containing data files.
        Sy_min (int): Minimum y index for spatial filtering.
        Sy_max (int): Maximum y index for spatial filtering.
        Sx_min (int): Minimum x index for spatial filtering.
        Sx_max (int): Maximum x index for spatial filtering.

    Returns:
        list: A list of file paths that meet the specified criteria.

    Raises:
        FileNotFoundError: If any specified file does not exist.
    """
    # Search for relevant files based on hour and minute
    dataset = search(hour, minute, data_path)
    H_0 = []
    
    for file in dataset:
        if not os.path.exists(file):
            continue  # Skip files that do not exist

        with Dataset(file, "r") as data:            
            lat = data["max_lat"][:]
            lon = data["max_lon"][:]

            # Loop through latitude and longitude pairs
            for lt, ln in zip(lat, lon):
                coords = to_yx(lt, ln)  # Get the indices
                if coords:  # Check if valid coordinates were returned
                    y, x = coords
                    # Check if coordinates fall within specified bounds
                    if Sy_min <= y <= Sy_max and Sx_min <= x <= Sx_max:
                        if file not in H_0:  # Avoid duplicates
                            H_0.append(file)

    return H_0



def compute_pc_x0(hour, minute, H0_data, spatial_filter_size, ymin, ymax, xmin, xmax):
    """
    Computes the core climatology (PC) for a specified hour and minute using data from H0.

    Args:
        hour (int): The hour to filter files.
        minute (int): The minute to filter files.
        H0_data (list of str): List of dates to construct filenames.
        spatial_filter_size (int): The size of the spatial filter to apply.
        ymin (int): The minimum y-coordinate for the region of interest.
        ymax (int): The maximum y-coordinate for the region of interest.
        xmin (int): The minimum x-coordinate for the region of interest.
        xmax (int): The maximum x-coordinate for the region of interest.

    Returns:
        np.ndarray: The computed core climatology (PC) array given x0.

    Raises:
        ValueError: If H0_data is empty or if spatial_filter_size is not positive.
    """
    if not H0_data:
        raise ValueError("H0_data cannot be empty.")
    
    if spatial_filter_size <= 0:
        raise ValueError("The spatial filter size must be a positive integer.")

    # Generate the dataset of filenames
    dataset = [f"{date}{hour:02d}{minute:02d}.nc" for date in H0_data]
    
    # Compute the core climatology (PC)
    pc = compute_pc(dataset=dataset, spatial_filter_size=spatial_filter_size, ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax)
    
    return pc


def squareId_to_S0(square_id):
    """
    Extracts the spatial coordinates from a square ID string.

    Args:
        square_id (str): A string formatted as "{Sy_min}_{Sx_min}_{Sy_max}_{Sx_max}".

    Returns:
        tuple: A tuple containing (Sy_min, Sy_max, Sx_min, Sx_max) as integers.

    Raises:
        ValueError: If the square_id format is incorrect.
    """
    result = re.search(r"^(\d+)_(\d+)_(\d+)_(\d+)$", square_id)
    
    if result:
        Sy_min = int(result.group(1))
        Sx_min = int(result.group(2))
        Sy_max = int(result.group(3))
        Sx_max = int(result.group(4))
        return Sy_min, Sy_max, Sx_min, Sx_max
    else:
        raise ValueError(f"Invalid square_id format: '{square_id}'")
    

def x0_from(lat, lon, lats, lons, raw_cores):
    """
    Extracts core values based on latitude and longitude.

    Args:
        lat (list or array-like): Latitude coordinates.
        lon (list or array-like): Longitude coordinates.
        raw_cores (np.ndarray): 2D array containing core values.

    Returns:
        list: A list of core values corresponding to the given lat/lon pairs,
              excluding masked values.

    Raises:
        ValueError: If lat and lon lengths do not match or if raw_cores is not a 2D array.
    """
    if len(lat) != len(lon):
        raise ValueError("Latitude and longitude lists must have the same length.")
    
    if raw_cores.ndim != 2:
        raise ValueError("raw_cores must be a 2D array.")

    x0 = []
    
    for lt, ln in zip(lat, lon):
        # Convert lat/lon to y/x indices
        indices = to_yx(lt, ln, lats, lons)
        if indices:  # Check if to_yx returned valid indices
            y, x = indices
            
            # Check if indices are within the bounds of raw_cores
            if 0 <= y < raw_cores.shape[0] and 0 <= x < raw_cores.shape[1]:
                core_val = raw_cores[y, x]
                if core_val is not np.ma.masked:  # Check for masked value
                    x0.append(core_val)
    
    return x0



def top(rank, ls, reverse=True):
    """
    Returns the top elements from a list, including duplicates, sorted in the specified order.

    Args:
        rank (int): The number of top elements to return.
        ls (list): The input list from which to extract elements.
        reverse (bool): If True, sort in descending order; if False, sort in ascending order.

    Returns:
        list: A list of the top elements, including duplicates, sorted in the specified order.

    Raises:
        ValueError: If rank is less than 1 or if the input list is empty.
    """
    if rank < 1:
        raise ValueError("Rank must be at least 1.")
    
    if not ls:  # Check if the input list is empty
        return []

    # Sort the list while keeping duplicates
    ls_sorted = sorted(ls, reverse=reverse)

    # Return the top 'rank' elements
    return ls_sorted[:rank]



def get_storm(binary_grid):
    """
    Identifies connected components (storms) in a binary grid and calculates their sizes.

    Args:
        binary_grid (np.ndarray): A binary array where non-zero values represent storm cells.

    Returns:
        dict: A dictionary containing:
            - 'labels': A labeled array where each storm has a unique label.
            - 'number_of_storms': The total number of identified storms.
            - 'size': A dictionary mapping each storm label to its size in square kilometers.
    
    Raises:
        ValueError: If the input is not a binary array or is empty.
    """
    if not isinstance(binary_grid, np.ndarray) or binary_grid.ndim != 2:
        raise ValueError("Input must be a 2D binary numpy array.")
    
    if binary_grid.size == 0:
        raise ValueError("Input binary grid cannot be empty.")

    # Label each connected grid cell
    labels, number_of_groups = measure.label(binary_grid, connectivity=2, return_num=True)

    # Taking the label of each identified group, excluding the background label (0)
    core_labels = np.unique(labels)
    core_labels = core_labels[core_labels != 0]

    # Dictionary to store the size of each storm
    dict_storm_size = {}
    for label in core_labels:
        num_cells = np.sum(labels == label)    
        dict_storm_size[str(label)] = num_cells * 9  # Assuming each cell represents 9 km²

    return {
        "labels": labels,
        "number_of_storms": number_of_groups,
        "size": dict_storm_size
    }


def get_x0_label(x0_power, cores, labels):
    """
    Retrieves the storm labels corresponding to a specific x0 power value.

    Args:
        x0_power (float/int): The specific power value to search for in the cores array.
        cores (np.ndarray): A 2D array where each cell represents a core's power.
        labels (np.ndarray): A 2D array where each cell contains the storm label.

    Returns:
        list: A list of storm labels associated with the specified x0 power value.

    Raises:
        ValueError: If the cores and labels arrays do not have the same shape.
    """
    # Ensure the cores and labels arrays have the same shape
    if cores.shape != labels.shape:
        raise ValueError("The cores and labels arrays must have the same shape.")

    # Find the indices of the cores that match the x0_power value
    indices = np.argwhere(cores == x0_power)
    
    # Extract the corresponding storm labels
    list_storm_label = labels[indices[:, 0], indices[:, 1]].tolist()
    
    return list_storm_label

