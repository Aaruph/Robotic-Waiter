import qrcode

# Data to be encoded
data = "Table no 0"

# Creating an instance of QRCode
qr = qrcode.QRCode(
    version=1,  # Controls the size of the QR Code
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)

# Adding data to the instance
qr.add_data(data)
qr.make(fit=True)

# Creating an image
img = qr.make_image(fill='black', back_color='white')

# Saving the image to a file
img.save("table_no_0.png")

print("QR code for 'Table no 0' generated and saved as table_no_0.png")
