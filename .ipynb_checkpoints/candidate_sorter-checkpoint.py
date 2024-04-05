from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import shutil
import base64

app = Flask(__name__)

# Directory containing the plots
plots_directory = 'candidates/'
# Directory where checked plots will be saved
checked_plots_directory = os.path.join(plots_directory, 'checked')
approved_plots_directory = 'follow_up_candidates/'

# Ensure the approved plots directory exists
os.makedirs(approved_plots_directory, exist_ok=True)
# Ensure the checked plots directory exists
os.makedirs(checked_plots_directory, exist_ok=True)

# List of plot file names, excluding those already checked
plot_files = [file for file in os.listdir(plots_directory) 
              if file.endswith(('png', 'jpg', 'jpeg')) and not os.path.exists(os.path.join(checked_plots_directory, file))]
print('plot files',plot_files)
plot_counter = 0

def get_plot_files():
    """Function to dynamically get the list of plot files."""
    return [file for file in os.listdir(plots_directory) 
            if file.endswith(('png', 'jpg', 'jpeg')) and not os.path.exists(os.path.join(checked_plots_directory, file))]

@app.route('/')
def index():
    global plot_counter
    plot_files = get_plot_files()
    if plot_counter < len(plot_files):
        plot_filename = plot_files[plot_counter]
        next_image_filename = plot_files[plot_counter + 1] if plot_counter + 1 < len(plot_files) else None
        return render_template('index.html', plot_filename=plot_filename, next_image_filename=next_image_filename)
    else:
        return "No more plots to display."

@app.route('/images/<filename>')
def serve_image(filename):
    print("Serving image:", filename)
    return send_from_directory(plots_directory, filename)

@app.route('/decision', methods=['POST'])
def decision():
    global plot_files, plot_counter
    choice = request.form['choice']
    if plot_counter < len(plot_files):
        if choice == "Yes":
            # Move the approved plot to the approved directory
            source_path = os.path.join(plots_directory, plot_files[plot_counter])
            destination_path = os.path.join(approved_plots_directory, plot_files[plot_counter])
            if not os.path.exists(destination_path):
                shutil.copy(source_path, destination_path)
        # Move the checked plot to the checked directory
        source_path = os.path.join(plots_directory, plot_files[plot_counter])
        destination_path = os.path.join(checked_plots_directory, plot_files[plot_counter])
        shutil.move(source_path, destination_path)
        plot_counter += 1
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
