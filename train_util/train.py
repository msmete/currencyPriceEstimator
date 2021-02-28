from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard


def train(model, X_train, y_train):
    es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
    mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)

    tb = TensorBoard('logs')

    history = model.fit(X_train, y_train, shuffle=True, epochs=10, callbacks=[mcp, tb], validation_split=0.2, verbose=1, batch_size=256)

    # history = model.fit(X_train, y_train, shuffle=True, epochs=200, callbacks=[es, rlr, mcp, tb], validation_split=0.2, verbose=1, batch_size=256)
    return model, history