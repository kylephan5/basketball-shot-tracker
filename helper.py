def close_to_hoop(rim_height, rim_width, center_x, center_y):
    width = rim_width[1] - rim_width[0]
    x1 = rim_width[0] + -1 * width
    x2 = rim_width[1] + 1 * width
    y1 = rim_height - 200
    y2 = rim_height + 200

    if x1 < center_x < x2 and y1 < center_y < y2:
        return True

    return False


def get_position_hoop(detections, model):
    rim_height = 0
    rim_width = (0, 0)
    for *box, conf, cls in detections:
        if model.names[int(cls)] == 'Basketball Hoop':
            x1, y1, x2, y2 = map(int, box)
            rim_height = y1
            rim_width = (x1, x2)
            break

    return rim_height, rim_width

